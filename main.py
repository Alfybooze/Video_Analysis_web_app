import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import base64
import uvicorn
import whisper
import google.generativeai as genai
from typing import Any, List
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from frame_transcript_matcher import (
    process_video_frames_with_transcript,
    get_batch_analysis_prompt,
    FrameBatch
)

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- FastAPI App ----------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable not set")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Configure OpenAI and Gemini
client = OpenAI(api_key=OPENAI_KEY)
genai.configure(api_key=GEMINI_API_KEY)

WORK_DIR = "video_jobs"
os.makedirs(WORK_DIR, exist_ok=True)

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)


class VideoRequest(BaseModel):
    video_url: str


def identify_platform(url: str) -> str:
    """Identify the video platform from URL"""
    url_lower = url.lower()
    
    if 'tiktok.com' in url_lower:
        return 'tiktok'
    elif 'twitter.com' in url_lower or 'x.com' in url_lower:
        return 'twitter'
    elif 'instagram.com' in url_lower:
        return 'instagram'
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif any(ext in url_lower for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
        return 'direct'
    else:
        return 'unknown'


def get_platform_specific_options(platform: str) -> list:
    """Get yt-dlp options specific to each platform"""
    base_options = ["yt-dlp", "--output", "-"]  # Output to stdout for streaming
    
    if platform == 'tiktok':
        return base_options + [
            "-f", "best[height<=720]/best",
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--referer", "https://www.tiktok.com/",
            "--no-playlist",
            "--ignore-config",
            "--geo-bypass",
            "--geo-bypass-country", "US",
            # Remove cookie option that's causing DPAPI errors
            "--no-check-certificate"
        ]
    elif platform == 'twitter':
        return base_options + [
            "-f", "best[height<=720]/best",
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--referer", "https://twitter.com/"
        ]
    elif platform == 'instagram':
        return base_options + [
            "-f", "best[height<=720]/best",
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--referer", "https://www.instagram.com/"
        ]
    else:
        return base_options + [
            "-f", "best[ext=mp4]/best",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "--no-check-certificate"
        ]


def get_enhanced_platform_analysis_settings(platform: str, video_duration: float = None) -> dict:
    """Get analysis settings optimized for each platform"""
    base_settings = {
        'tiktok': {
            'max_duration': 300,
            'base_interval': 3,
            'max_frames': 30,
            'audio_priority': True,
            'fps_filter': 'fps=1/3',
            'scene_change_detection': True,
            'min_frames': 8
        },
        'twitter': {
            'max_duration': 140,
            'base_interval': 3,
            'max_frames': 30,
            'audio_priority': True,
            'fps_filter': 'fps=1/3',
            'scene_change_detection': True,
            'min_frames': 6
        },
        'instagram': {
            'max_duration': 300,
            'base_interval': 10,
            'max_frames': 30,
            'audio_priority': True,
            'fps_filter': 'fps=1/5',
            'scene_change_detection': True,
            'min_frames': 8
        },
        'youtube': {
            'max_duration': 1800,
            'base_interval': 10,
            'max_frames': 30,
            'audio_priority': True,
            'fps_filter': 'fps=1/15',
            'scene_change_detection': False,
            'min_frames': 8
        },
        'default': {
            'max_duration': 600,
            'base_interval': 8,
            'max_frames': 30,
            'audio_priority': True,
            'fps_filter': 'fps=1/8',
            'scene_change_detection': True,
            'min_frames': 6
        }
    }
    
    settings = base_settings.get(platform, base_settings['default']).copy()
    
    if video_duration:
        if video_duration <= 30:
            settings['base_interval'] = min(settings['base_interval'], 2)
            settings['fps_filter'] = 'fps=1/2'
        elif video_duration > 600:
            settings['max_frames'] = min(25, int(video_duration / 30))
    
    return settings


async def download_video_async(url: str, platform: str, output_path: str) -> bool:
    """Download video to local file asynchronously"""
    logger.info(f"Starting async video download for {platform}...")
    
    try:
        # Get platform-specific options but change output to file
        cmd_options = ["yt-dlp", "-o", output_path, "-f", "best[height<=720]/best"]
        
        if platform == 'tiktok':
            cmd_options.extend([
                "--no-check-certificate",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "--referer", "https://www.tiktok.com/",
                "--no-playlist",
                "--geo-bypass"
            ])
        
        cmd_options.append(url)
        
        # Run subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd_options,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Video downloaded successfully ({file_size:.2f} MB)")
            return True
        else:
            logger.error(f"Download failed: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"Video download error: {e}")
        return False


async def extract_audio_async(video_path: str, audio_path: str, max_duration: int) -> bool:
    """Extract audio from video asynchronously"""
    logger.info("Starting async audio extraction...")
    start_time = time.time()
    
    try:
        audio_cmd = [
            "ffmpeg", "-i", video_path,
            "-t", str(max_duration),
            "-q:a", "0",
            "-map", "a",
            "-ac", "1",
            "-ar", "16000",
            audio_path, "-y"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *audio_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0 and os.path.exists(audio_path):
            logger.info(f"Audio extracted successfully in {elapsed:.2f}s")
            return True
        else:
            logger.warning(f"Audio extraction failed after {elapsed:.2f}s")
            return False
            
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return False


async def extract_frames_async(video_path: str, frames_dir: str, settings: dict) -> list:
    """Extract frames from video asynchronously"""
    logger.info("Starting async frame extraction...")
    start_time = time.time()
    
    try:
        # Build frame extraction command
        if settings.get('scene_change_detection', False):
            video_filter = f"select='gte(t*{1/settings['base_interval']},n)+gt(scene,0.3)',scale=-1:720"
        else:
            video_filter = f"fps={1/settings['base_interval']},scale=-1:720"
        
        frame_cmd = [
            "ffmpeg", "-i", video_path,
            "-t", str(settings['max_duration']),
            "-vf", video_filter,
            "-vsync", "vfr",
            "-frame_pts", "1",
            "-threads", "4",
            "-preset", "ultrafast",
            "-q:v", "2",
            f"{frames_dir}/frame_%04d.jpg"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *frame_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            frame_files = frame_files[:settings['max_frames']]
            logger.info(f"Extracted {len(frame_files)} frames in {elapsed:.2f}s")
            return frame_files
        else:
            logger.error(f"Frame extraction failed after {elapsed:.2f}s")
            return []
            
    except Exception as e:
        logger.error(f"Frame extraction error: {e}")
        return []


def transcribe_audio_sync(audio_path: str, platform: str) -> tuple[str, str]:
    """Transcribe audio synchronously (runs in thread pool)"""
    logger.info("Starting audio transcription...")
    start_time = time.time()
    
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            return "", ""
        
        model = whisper.load_model("base")
        whisper_result = model.transcribe(
            audio_path,
            word_timestamps=True,
            task="transcribe"
        )
        
        transcript_with_timestamps = []
        transcript_text_only = ""
        
        if "segments" in whisper_result:
            for segment in whisper_result["segments"]:
                start_time_seg = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                if text:
                    start_formatted = f"{int(start_time_seg//60):02d}:{int(start_time_seg%60):02d}"
                    end_formatted = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                    transcript_with_timestamps.append(f"[{start_formatted}-{end_formatted}] {text}")
                    transcript_text_only += text + " "
        
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f}s")
        
        return transcript_text_only.strip(), "\n".join(transcript_with_timestamps)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "", ""


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_platform_optimized_vision_prompt(platform: str, frame_count: int) -> str:
    """Generate optimized prompts for vision model"""
    platform_contexts = {
        'tiktok': {
            'focus': "viral trends, text overlays, effects, transitions, gaming, music/dance, comedy skits",
            'style': "fast-paced, engaging, youth-oriented content",
            'key_elements': "hashtag-worthy moments, sound sync, visual effects, trending topics"
        },
        'twitter': {
            'focus': "news, reactions, viral moments, discussions, breaking events",
            'style': "conversational, immediate, topical content",
            'key_elements': "key quotes, reactions, context clues, trending topics"
        },
        'instagram': {
            'focus': "lifestyle, aesthetics, stories, reels, visual appeal",
            'style': "polished, aspirational, visually appealing",
            'key_elements': "composition, lighting, brand elements, lifestyle moments"
        },
        'youtube': {
            'focus': "educational content, tutorials, entertainment, storytelling",
            'style': "structured, informative, engaging long-form content",
            'key_elements': "key teaching moments, demonstrations, narrative flow"
        }
    }
    
    context = platform_contexts.get(platform, {
        'focus': "general video content",
        'style': "varied content types",
        'key_elements': "main visual elements"
    })
    
    return f"""You are analyzing {frame_count} sequential frames from a {platform.upper()} video.

PLATFORM CONTEXT: Focus on {context['focus']}. This is {context['style']}.

ANALYSIS INSTRUCTIONS:
1. For each frame, identify: {context['key_elements']}
2. If it's gaming content: Name the game, platform (mobile/PC/console), UI elements
3. If it's educational: Identify the subject area and teaching method
4. If it's entertainment: Note the format (comedy, music, dance, etc.)
5. Track visual progression and scene changes between frames

RESPONSE FORMAT: 
Provide Frame 1, Frame 2, etc. with 1-2 concise sentences each.
Focus on elements that help understand the video's purpose and appeal to {platform} users."""


async def generate_batch_descriptions_async(
    batches: List[FrameBatch],
    frames_dir: str,
    platform: str
) -> List[dict]:
    """
    Generate descriptions for each batch of frames with transcript context
    
    Args:
        batches: List of FrameBatch objects
        frames_dir: Directory containing extracted frames
        platform: Video platform
    
    Returns:
        List of batch descriptions with metadata
    """
    logger.info(f"Generating descriptions for {len(batches)} batches...")
    batch_results = []
    
    for batch in batches:
        try:
            # Prepare content with frames and prompt
            prompt = get_batch_analysis_prompt(batch, platform)
            
            content: list[Any] = [{"type": "text", "text": prompt}]
            
            # Add frames to the request
            for frame in batch.frames:
                frame_path = os.path.join(frames_dir, frame.filename)
                
                if not os.path.exists(frame_path):
                    logger.warning(f"Frame file not found: {frame_path}")
                    continue
                
                try:
                    base64_image = encode_image(frame_path)
                    content.extend([
                        {"type": "text", "text": f"\nFrame {frame.frame_number} (~{int(frame.timestamp)}s):"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ])
                except Exception as e:
                    logger.error(f"Error encoding frame {frame.filename}: {e}")
                    continue
            
            messages: list[Any] = [{"role": "user", "content": content}]
            
            # Call vision API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.3
                )
            )
            
            description = response.choices[0].message.content
            
            batch_result = {
                "batch_number": batch.batch_number,
                "frame_count": len(batch.frames),
                "time_range": f"{int(batch.start_time)}s - {int(batch.end_time)}s",
                "transcript": batch.combined_transcript,
                "description": description,
                "frames": [
                    {
                        "number": f.frame_number,
                        "filename": f.filename,
                        "timestamp": round(f.timestamp, 2)
                    }
                    for f in batch.frames
                ]
            }
            
            batch_results.append(batch_result)
            logger.info(f"Batch {batch.batch_number} description generated ({len(batch.frames)} frames)")
            
        except Exception as e:
            logger.error(f"Error generating description for batch {batch.batch_number}: {e}")
            batch_results.append({
                "batch_number": batch.batch_number,
                "error": str(e),
                "frame_count": len(batch.frames)
            })
    
    return batch_results


async def generate_frame_captions_async(frames_dir: str, frame_files: list, platform: str) -> list:
    """Generate captions for frames asynchronously"""
    logger.info("Starting async frame caption generation...")
    start_time = time.time()
    
    frame_summaries = []
    platform_prompt = get_platform_optimized_vision_prompt(platform, len(frame_files))
    
    content: list[Any] = [{"type": "text", "text": platform_prompt}]
    
    for idx, fname in enumerate(frame_files, start=1):
        frame_path = os.path.join(frames_dir, fname)
        try:
            base64_image = encode_image(frame_path)
            content.extend([
                {"type": "text", "text": f"\nFrame {idx}:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                }
            ])
        except Exception as e:
            logger.error(f"Error reading {fname}: {e}")
            continue
    
    messages: list[Any] = [{"role": "user", "content": content}]
    
    try:
        # Run OpenAI API call in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
        )
        
        captions_text = response.choices[0].message.content
        
        if captions_text:
            lines = captions_text.splitlines()
            for line in lines:
                line = line.strip()
                if line and ("frame" in line.lower() or any(char.isdigit() for char in line[:10])):
                    frame_summaries.append(line)
        
        elapsed = time.time() - start_time
        logger.info(f"Generated {len(frame_summaries)} captions in {elapsed:.2f}s")
        
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        for idx in range(1, len(frame_files) + 1):
            frame_summaries.append(f"Frame {idx}: {platform} video content")
    
    return frame_summaries


async def generate_summary_async(transcript: str, transcript_timestamps: str, 
                                frame_summaries: list, platform: str) -> str:
    """Generate final summary asynchronously"""
    logger.info("Starting async summary generation...")
    start_time = time.time()
    
    platform_summary_context = {
        'tiktok': "This is a TikTok video. Summarize with a focus on short-form engagement: social trends, gaming, music/audio elements, viral hooks, and visual effects.",
        'twitter': "This is a Twitter video. Summarize with a focus on news, current events, viral reactions, or discussions that trend on the platform.",
        'instagram': "This is an Instagram video. Summarize with a focus on lifestyle, aesthetics, reels/stories, visual creativity, and shareability.",
        'youtube': "This is a YouTube video. Summarize with a focus on educational material, tutorials, entertainment, gaming, or documentary-style storytelling.",
        'default': "This is a video. Provide a clear and engaging summary of what the content is about."
    }
    
    if transcript:
        combined_context = f"""
{platform_summary_context.get(platform, platform_summary_context['default'])}

AUDIO TRANSCRIPT WITH TIMESTAMPS:
{transcript_timestamps}

VISUAL SNAPSHOTS (enhanced frame analysis with {len(frame_summaries)} frames):
{chr(10).join(frame_summaries)}

TASK:
Provide a structured, detailed summary that combines both the audio and visual elements. 

1. Identify what the video is about (e.g., gaming → name the game, anime → name the anime, educational → name the topic, music → name the song/artist, Dance → name the dance).
2. Highlight all the key points or moments using the EXACT TIMESTAMPS from the transcribed audio above (e.g., "At [00:15-00:25], the speaker discusses..." or "Between [01:30-01:45], we see...").
3. Reference specific timestamps when describing visual elements that align with the audio.
4. Explain the key benefits of watching the {platform.upper()} video (trends, entertainment, learning value, cultural relevance, etc.).
5. Write in a clear, audience-friendly way (easy to read, short paragraphs, avoid jargon).
"""
    else:
        combined_context = f"""
{platform_summary_context.get(platform, platform_summary_context['default'])}

VISUAL SNAPSHOTS (enhanced frame analysis with {len(frame_summaries)} frames):
{chr(10).join(frame_summaries)}

TASK:
Provide a structured, detailed summary of this video using only the enhanced visual analysis.

1. Identify the type of content (music, dance, anime, gaming, tutorial, lifestyle, etc.).
2. Mention any recognizable people, characters, brands (if not famous, classify as upcoming/independent).
3. Highlight visual trends, styles, and effects that make it engaging for {platform.upper()} users.
4. Explain the likely audience appeal (why someone would watch/share it).
"""
    
    try:
        loop = asyncio.get_event_loop()
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await loop.run_in_executor(
            executor,
            lambda: model.generate_content(combined_context)
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Summary generated in {elapsed:.2f}s")
        
        return response.text
        
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.get("/")
def home():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")


@app.post("/summarize_video")
async def summarize_video(req: VideoRequest):
    """Main async endpoint for video summarization with parallel processing"""
    job_dir = None
    overall_start = time.time()
    
    logger.info(f"=== Starting async video summarization for {req.video_url} ===")
    
    try:
        # Step 1: Identify platform and setup
        platform = identify_platform(req.video_url)
        logger.info(f"Detected platform: {platform}")
        
        job_dir = os.path.join(WORK_DIR, f"job_{int(time.time())}")
        os.makedirs(job_dir, exist_ok=True)
        
        video_path = f"{job_dir}/video.mp4"
        audio_path = f"{job_dir}/audio.mp3"
        frames_dir = f"{job_dir}/frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Step 2: Download video first
        download_success = await download_video_async(req.video_url, platform, video_path)
        
        if not download_success:
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        # Get video duration
        duration_result = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await duration_result.communicate()
        
        video_duration = 0
        try:
            import json
            data = json.loads(stdout.decode())
            video_duration = float(data.get("format", {}).get("duration", 0))
            logger.info(f"Video duration: {video_duration:.1f}s")
        except:
            logger.warning("Could not determine video duration")
        
        analysis_settings = get_enhanced_platform_analysis_settings(platform, video_duration)
        logger.info(f"Using settings: {analysis_settings}")
        
        # Step 3: Run audio extraction and frame extraction IN PARALLEL
        logger.info("=== PARALLEL PROCESSING: Audio + Frames ===")
        parallel_start = time.time()
        
        audio_task = extract_audio_async(video_path, audio_path, analysis_settings['max_duration'])
        frames_task = extract_frames_async(video_path, frames_dir, analysis_settings)
        
        # Wait for both to complete
        audio_success, frame_files = await asyncio.gather(audio_task, frames_task)
        
        parallel_elapsed = time.time() - parallel_start
        logger.info(f"=== PARALLEL PROCESSING COMPLETED in {parallel_elapsed:.2f}s ===")
        
        # Step 4: Run transcription and caption generation IN PARALLEL
        logger.info("=== PARALLEL PROCESSING: Transcription + Captions ===")
        ai_start = time.time()
        
        # Transcription in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        transcription_task = loop.run_in_executor(
            executor,
            transcribe_audio_sync,
            audio_path,
            platform
        )
        
        # Caption generation (API call)
        captions_task = generate_frame_captions_async(frames_dir, frame_files, platform)
        
        # Wait for both to complete
        (transcript, transcript_timestamps), frame_summaries = await asyncio.gather(
            transcription_task,
            captions_task
        )
        
        ai_elapsed = time.time() - ai_start
        logger.info(f"=== AI PROCESSING COMPLETED in {ai_elapsed:.2f}s ===")
        
        # Step 5: Generate final summary
        summary = await generate_summary_async(
            transcript,
            transcript_timestamps,
            frame_summaries,
            platform
        )
        
        # Cleanup
        shutil.rmtree(job_dir)
        
        overall_elapsed = time.time() - overall_start
        logger.info(f"=== TOTAL PROCESSING TIME: {overall_elapsed:.2f}s ===")
        
        return {
            "summary": summary,
            "platform": platform,
            "transcript_excerpt": transcript[:500] if transcript else f"No transcript (music/{platform} video)",
            "frames_analyzed": len(frame_files),
            "has_audio_transcript": bool(transcript.strip()),
            "video_duration": video_duration,
            "processing_time": {
                "total": round(overall_elapsed, 2),
                "parallel_extraction": round(parallel_elapsed, 2),
                "ai_processing": round(ai_elapsed, 2)
            },
            "performance_improvement": "Parallel processing enabled",
            "vision_api_used": "OpenAI GPT-4 Vision",
            "enhancements": [
                "Async parallel processing",
                "Simultaneous audio + frame extraction",
                "Concurrent transcription + caption generation",
                "Local video download for reliability"
            ]
        }
        
    except Exception as e:
        logger.exception("Error during async summarization")
        if job_dir and os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise


@app.post("/analyze_video_with_batches")
async def analyze_video_with_batches(req: VideoRequest):
    """
    New endpoint: Extract frames, match to transcript, and generate batch descriptions
    This approach reduces AI API load by batching frames with their corresponding transcript
    """
    job_dir = None
    overall_start = time.time()
    
    logger.info(f"=== Starting batch analysis for {req.video_url} ===")
    
    try:
        # Step 1: Identify platform and setup
        platform = identify_platform(req.video_url)
        logger.info(f"Detected platform: {platform}")
        
        job_dir = os.path.join(WORK_DIR, f"job_{int(time.time())}")
        os.makedirs(job_dir, exist_ok=True)
        
        video_path = f"{job_dir}/video.mp4"
        audio_path = f"{job_dir}/audio.mp3"
        frames_dir = f"{job_dir}/frames"
        metadata_dir = f"{job_dir}/batch_metadata"
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Step 2: Download video
        download_success = await download_video_async(req.video_url, platform, video_path)
        
        if not download_success:
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        # Step 3: Get video duration
        duration_result = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await duration_result.communicate()
        
        video_duration = 0
        try:
            import json as json_module
            data = json_module.loads(stdout.decode())
            video_duration = float(data.get("format", {}).get("duration", 0))
            logger.info(f"Video duration: {video_duration:.1f}s")
        except:
            logger.warning("Could not determine video duration")
        
        analysis_settings = get_enhanced_platform_analysis_settings(platform, video_duration)
        
        # Step 4: Extract audio and transcribe
        logger.info("Extracting and transcribing audio...")
        audio_success = await extract_audio_async(video_path, audio_path, analysis_settings['max_duration'])
        
        if not audio_success:
            raise HTTPException(status_code=400, detail="Failed to extract audio")
        
        # Transcribe audio
        loop = asyncio.get_event_loop()
        model = whisper.load_model("base")
        whisper_result = await loop.run_in_executor(
            executor,
            lambda: model.transcribe(audio_path, word_timestamps=True, task="transcribe")
        )
        
        # Step 5: Process frames with transcript matching
        logger.info("Processing frames with transcript matching (2 FPS)...")
        batches, stats = await process_video_frames_with_transcript(
            video_path=video_path,
            whisper_result=whisper_result,
            frames_dir=frames_dir,
            metadata_dir=metadata_dir,
            fps=2.0,  # 2 frames per second as requested
            max_duration=analysis_settings['max_duration'],
            batch_size=10  # 10 frames per batch
        )
        
        if not batches:
            raise HTTPException(status_code=400, detail="Failed to create frame batches")
        
        logger.info(f"Created {len(batches)} batches with frame-transcript matching")
        
        # Step 6: Generate descriptions for each batch
        logger.info("Generating descriptions for each batch...")
        batch_descriptions = await generate_batch_descriptions_async(
            batches=batches,
            frames_dir=frames_dir,
            platform=platform
        )
        
        # Step 7: Create comprehensive summary
        logger.info("Creating comprehensive summary...")
        
        summary_content = f"""
BATCH-BASED VIDEO ANALYSIS REPORT
==================================

Video Platform: {platform.upper()}
Total Duration: {video_duration:.1f} seconds
Frame Extraction: 2 FPS
Total Batches: {len(batches)}
Frames per Batch: 10
Total Frames Analyzed: {stats.get('total_frames', 'N/A')}

BATCH SUMMARIES:
"""
        
        for batch_desc in batch_descriptions:
            if "error" not in batch_desc:
                summary_content += f"""
Batch {batch_desc['batch_number']} ({batch_desc['time_range']})
{'-' * 40}
Frames: {batch_desc['frame_count']}
Transcript: {batch_desc['transcript']}
Description:
{batch_desc['description']}

"""
            else:
                summary_content += f"Batch {batch_desc['batch_number']}: Error - {batch_desc['error']}\n"
        
        # Cleanup
        shutil.rmtree(job_dir)
        
        overall_elapsed = time.time() - overall_start
        logger.info(f"=== TOTAL PROCESSING TIME: {overall_elapsed:.2f}s ===")
        
        return {
            "status": "success",
            "platform": platform,
            "video_duration": video_duration,
            "total_batches": len(batches),
            "total_frames_analyzed": stats.get('total_frames', 0),
            "frames_per_batch": 10,
            "extraction_fps": 2.0,
            "batch_descriptions": batch_descriptions,
            "comprehensive_summary": summary_content,
            "processing_stats": stats,
            "processing_time_seconds": round(overall_elapsed, 2),
            "methodology": "Frame extraction at 2 FPS with transcript matching and batch-based AI analysis"
        }
        
    except Exception as e:
        logger.exception("Error during batch analysis")
        if job_dir and os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "processing_mode": "Async with Parallel Execution + Batch Analysis",
        "supported_platforms": ["TikTok", "Twitter/X", "Instagram", "YouTube", "Direct URLs"],
        "yt_dlp_available": shutil.which("yt-dlp") is not None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
        "features": [
            "Parallel audio + frame extraction",
            "Concurrent transcription + caption generation",
            "Local video download",
            "Adaptive frame sampling",
            "Platform-optimized analysis",
            "NEW: Frame-transcript matching at 2 FPS",
            "NEW: Batch-based AI analysis (reduces API calls)",
            "NEW: Frame-accurate descriptions with transcript context"
        ],
        "new_endpoints": {
            "analyze_video_with_batches": "POST /analyze_video_with_batches - Batch analysis with frame-to-transcript matching",
            "summarize_video": "POST /summarize_video - Original full summarization"
        }
    }


if __name__ == "__main__":
    logger.info("Starting ASYNC FastAPI server with parallel processing...")
    uvicorn.run(app, host="0.0.0.0", port=8000)