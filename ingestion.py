"""
Video Ingestion Module
Handles video downloading, metadata extraction, and initial setup
"""
import os
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import json
import ffmpeg
from urllib.parse import urlparse
import requests
from loguru import logger

from settings import settings
from timeline import MasterTimeline


class VideoIngestionError(Exception):
    """Custom exception for video ingestion errors"""
    pass


class VideoIngestion:
    """
    Video Ingestion Pipeline
    - Downloads or validates video URL
    - Extracts metadata (duration, resolution, fps, codec)
    - Creates master timeline
    - Prepares storage structure
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir or settings.VIDEO_TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_video_id(self, video_url: str) -> str:
        """Generate unique video ID from URL"""
        return hashlib.sha256(video_url.encode()).hexdigest()[:16]
    
    def _download_video(self, video_url: str, output_path: Path) -> None:
        """Download video from URL"""
        logger.info(f"Downloading video from {video_url}")
        
        try:
            # Check if it's a local file path
            if os.path.exists(video_url):
                # If local file, just copy or symlink
                import shutil
                shutil.copy2(video_url, output_path)
                logger.info(f"Copied local video to {output_path}")
                return
            
            # Otherwise, download from URL
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Video downloaded to {output_path}")
            
        except Exception as e:
            raise VideoIngestionError(f"Failed to download video: {str(e)}")
    
    def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe"""
        logger.info(f"Extracting metadata from {video_path}")
        
        try:
            probe = ffmpeg.probe(str(video_path))
            
            # Get video stream
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            # Get audio stream
            audio_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'audio'),
                None
            )
            
            if not video_stream:
                raise VideoIngestionError("No video stream found in file")
            
            # Extract duration
            duration = float(probe['format'].get('duration', 0))
            
            if duration > settings.MAX_VIDEO_DURATION:
                raise VideoIngestionError(
                    f"Video duration {duration}s exceeds maximum {settings.MAX_VIDEO_DURATION}s"
                )
            
            metadata = {
                "duration": duration,
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "fps": eval(video_stream.get('r_frame_rate', '0/1')),  # Convert fraction to float
                "codec": video_stream.get('codec_name'),
                "format": probe['format'].get('format_name'),
                "bitrate": int(probe['format'].get('bit_rate', 0)),
                "has_audio": audio_stream is not None,
                "audio_codec": audio_stream.get('codec_name') if audio_stream else None,
                "file_size": int(probe['format'].get('size', 0))
            }
            
            logger.info(f"Metadata extracted: {json.dumps(metadata, indent=2)}")
            return metadata
            
        except ffmpeg.Error as e:
            raise VideoIngestionError(f"FFprobe error: {e.stderr.decode()}")
        except Exception as e:
            raise VideoIngestionError(f"Failed to extract metadata: {str(e)}")
    
    def ingest(self, video_url: str, force_download: bool = False) -> tuple[str, Path, MasterTimeline]:
        """
        Main ingestion pipeline
        
        Args:
            video_url: URL or path to video file
            force_download: Force re-download even if file exists
        
        Returns:
            tuple: (video_id, video_path, master_timeline)
        """
        logger.info(f"Starting video ingestion for: {video_url}")
        
        # Generate video ID
        video_id = self._generate_video_id(video_url)
        
        # Prepare paths
        video_filename = f"{video_id}.mp4"
        video_path = self.temp_dir / video_filename
        
        # Download or validate video
        if not video_path.exists() or force_download:
            self._download_video(video_url, video_path)
        else:
            logger.info(f"Using existing video at {video_path}")
        
        # Extract metadata
        metadata = self._extract_metadata(video_path)
        
        # Create master timeline
        timeline = MasterTimeline(
            video_id=video_id,
            duration_seconds=metadata['duration'],
            metadata={
                **metadata,
                "source_url": video_url,
                "local_path": str(video_path)
            }
        )
        
        logger.info(f"Video ingestion complete. Timeline: {timeline}")
        
        return video_id, video_path, timeline
    
    def validate_video(self, video_path: Path) -> bool:
        """Validate that video file is playable"""
        try:
            ffmpeg.probe(str(video_path))
            return True
        except:
            return False


# Convenience function
def ingest_video(video_url: str, **kwargs) -> tuple[str, Path, MasterTimeline]:
    """
    Convenience function for video ingestion
    
    Usage:
        video_id, video_path, timeline = ingest_video("https://example.com/video.mp4")
    """
    ingestion = VideoIngestion()
    return ingestion.ingest(video_url, **kwargs)