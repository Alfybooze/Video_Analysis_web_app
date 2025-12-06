import os
import asyncio
import logging
import base64
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import subprocess
import time

logger = logging.getLogger(__name__)

@dataclass
class FrameBatch:
    """Represents a batch of frames with their corresponding transcript"""
    batch_number: int
    start_time: float
    end_time: float
    frame_paths: List[str]
    transcript_segments: List[Dict[str, Any]]
    frame_times: List[float]
    
    def get_combined_transcript(self) -> str:
        """Get combined transcript text for this batch"""
        if not self.transcript_segments:
            return "No speech detected in this time range"
        
        combined_text = []
        for segment in self.transcript_segments:
            if 'text' in segment:
                combined_text.append(segment['text'].strip())
        
        return ' '.join(combined_text) if combined_text else "No speech detected in this time range"

async def extract_frames_at_fps(
    video_path: str,
    frames_dir: str,
    fps: float = 2.0,
    max_duration: int = 600
) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """
    Extract frames from video at specified FPS
    Returns: (frame_paths, frame_times, stats)
    """
    try:
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video duration first
        duration_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path
        ]
        
        duration_result = await asyncio.create_subprocess_exec(
            *duration_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await duration_result.communicate()
        
        video_duration = 0
        try:
            data = json.loads(stdout.decode())
            video_duration = float(data.get("format", {}).get("duration", 0))
        except:
            logger.warning("Could not determine video duration, using max_duration")
            video_duration = max_duration
        
        # Limit duration
        actual_duration = min(video_duration, max_duration)
        
        # Calculate total frames needed
        total_frames = int(actual_duration * fps)
        
        # Extract frames using ffmpeg
        frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
        
        extract_cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-t", str(actual_duration),
            "-q:v", "2",  # High quality
            frame_pattern
        ]
        
        logger.info(f"Extracting {total_frames} frames at {fps} FPS from {actual_duration}s video")
        
        extract_process = await asyncio.create_subprocess_exec(
            *extract_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await extract_process.communicate()
        
        if extract_process.returncode != 0:
            logger.error(f"FFmpeg frame extraction failed: {stderr.decode()}")
            return [], [], {}
        
        # Collect frame paths and times
        frame_paths = []
        frame_times = []
        
        for i in range(total_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i+1:04d}.jpg")
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
                frame_times.append(i / fps)  # Time in seconds
        
        stats = {
            'total_frames': len(frame_paths),
            'fps': fps,
            'duration': actual_duration,
            'extraction_time': time.time()
        }
        
        logger.info(f"Successfully extracted {len(frame_paths)} frames")
        return frame_paths, frame_times, stats
        
    except Exception as e:
        logger.exception("Error extracting frames")
        return [], [], {}

def match_frames_to_transcript(
    frame_times: List[float],
    whisper_result: Dict[str, Any],
    time_window: float = 2.0
) -> List[List[Dict[str, Any]]]:
    """
    Match frames to transcript segments based on timing
    Returns: List of transcript segments for each frame
    """
    if not whisper_result or 'segments' not in whisper_result:
        return [[] for _ in frame_times]
    
    segments = whisper_result['segments']
    frame_transcripts = []
    
    for frame_time in frame_times:
        matching_segments = []
        
        for segment in segments:
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', segment_start)
            
            # Check if frame time falls within segment time + window
            if (segment_start - time_window) <= frame_time <= (segment_end + time_window):
                matching_segments.append(segment)
        
        frame_transcripts.append(matching_segments)
    
    return frame_transcripts

def create_frame_batches(
    frame_paths: List[str],
    frame_times: List[float],
    frame_transcripts: List[List[Dict[str, Any]]],
    batch_size: int = 10
) -> List[FrameBatch]:
    """
    Group frames into batches for efficient processing
    """
    batches = []
    total_frames = len(frame_paths)
    
    for i in range(0, total_frames, batch_size):
        batch_frames = frame_paths[i:i+batch_size]
        batch_times = frame_times[i:i+batch_size]
        batch_transcripts = frame_transcripts[i:i+batch_size]
        
        # Flatten transcripts for this batch
        all_transcripts = []
        for trans_list in batch_transcripts:
            all_transcripts.extend(trans_list)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_transcripts = []
        for trans in all_transcripts:
            trans_id = trans.get('id', str(trans))
            if trans_id not in seen_ids:
                seen_ids.add(trans_id)
                unique_transcripts.append(trans)
        
        if batch_times:
            start_time = batch_times[0]
            end_time = batch_times[-1]
        else:
            start_time = 0
            end_time = 0
        
        batch = FrameBatch(
            batch_number=len(batches) + 1,
            start_time=start_time,
            end_time=end_time,
            frame_paths=batch_frames,
            transcript_segments=unique_transcripts,
            frame_times=batch_times
        )
        
        batches.append(batch)
    
    return batches

async def process_video_frames_with_transcript(
    video_path: str,
    whisper_result: Dict[str, Any],
    frames_dir: str,
    metadata_dir: str,
    fps: float = 2.0,
    max_duration: int = 600,
    batch_size: int = 10
) -> Tuple[List[FrameBatch], Dict[str, Any]]:
    """
    Main function to process video frames with transcript matching
    Returns: (batches, stats)
    """
    try:
        logger.info(f"Starting frame extraction and transcript matching...")
        
        # Step 1: Extract frames
        frame_paths, frame_times, extract_stats = await extract_frames_at_fps(
            video_path, frames_dir, fps, max_duration
        )
        
        if not frame_paths:
            logger.error("No frames extracted")
            return [], {}
        
        # Step 2: Match frames to transcript
        logger.info("Matching frames to transcript segments...")
        frame_transcripts = match_frames_to_transcript(frame_times, whisper_result)
        
        # Step 3: Create batches
        logger.info(f"Creating batches of {batch_size} frames each...")
        batches = create_frame_batches(frame_paths, frame_times, frame_transcripts, batch_size)
        
        # Step 4: Save metadata
        os.makedirs(metadata_dir, exist_ok=True)
        
        metadata = {
            'total_frames': len(frame_paths),
            'fps': fps,
            'batch_size': batch_size,
            'total_batches': len(batches),
            'frame_times': frame_times,
            'extraction_stats': extract_stats
        }
        
        metadata_path = os.path.join(metadata_dir, 'batch_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create batch-specific metadata
        for batch in batches:
            batch_metadata = {
                'batch_number': batch.batch_number,
                'time_range': f"{batch.start_time:.1f}s - {batch.end_time:.1f}s",
                'frame_count': len(batch.frame_paths),
                'transcript': batch.get_combined_transcript(),
                'frame_times': batch.frame_times
            }
            
            batch_meta_path = os.path.join(metadata_dir, f'batch_{batch.batch_number:03d}_metadata.json')
            with open(batch_meta_path, 'w') as f:
                json.dump(batch_metadata, f, indent=2)
        
        stats = {
            'total_frames': len(frame_paths),
            'total_batches': len(batches),
            'fps': fps,
            'batch_size': batch_size,
            'extraction_time': extract_stats.get('extraction_time', 0)
        }
        
        logger.info(f"Successfully created {len(batches)} batches from {len(frame_paths)} frames")
        return batches, stats
        
    except Exception as e:
        logger.exception("Error in process_video_frames_with_transcript")
        return [], {}

def get_batch_analysis_prompt(platform: str, batch_info: Dict[str, Any]) -> str:
    """
    Generate a prompt for analyzing a batch of frames with transcript context
    """
    time_range = batch_info.get('time_range', 'Unknown')
    transcript = batch_info.get('transcript', 'No transcript available')
    frame_count = batch_info.get('frame_count', 0)
    
    prompt = f"""Analyze this batch of {frame_count} video frames from {time_range} on {platform}.

TRANSCRIPT CONTEXT:
{transcript}

Please provide a detailed analysis that includes:

1. **Visual Description**: Describe what's happening visually in these frames
2. **Context Connection**: How do the visuals relate to the transcript content?
3. **Key Moments**: Identify any significant actions, expressions, or visual elements
4. **Platform Context**: Consider the typical content style and audience expectations for {platform}
5. **Overall Summary**: Provide a concise summary of what this batch represents

Be specific and detailed in your analysis. Focus on actionable insights that would help understand the video content.

Format your response in clear sections with headers."""

    return prompt