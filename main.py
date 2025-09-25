import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import requests
import shutil
import base64
import uvicorn
import whisper
import google.generativeai as genai
import re
from urllib.parse import urlparse
from typing import Union, Any
from openai import OpenAI

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()          # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# ---------------- FastAPI App ----------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
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
genai.configure(api_key=GEMINI_API_KEY) # type: ignore

WORK_DIR = "video_jobs"
os.makedirs(WORK_DIR, exist_ok=True)


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
    base_options = ["yt-dlp", "--get-url"]
    
    if platform == 'tiktok':
        # TikTok-specific options
        return base_options + [
            "-f", "best[height<=720]/best",  # Prefer 720p or lower for faster processing
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ]
    elif platform == 'twitter':
        # Twitter-specific options
        return base_options + [
            "-f", "best[height<=720]/best",
            "--no-check-certificate"
        ]
    elif platform == 'instagram':
        # Instagram-specific options
        return base_options + [
            "-f", "best[height<=720]/best",
            "--no-check-certificate"
        ]
    else:
        # Default options for other platforms
        return base_options + ["-f", "best[ext=mp4]/best"]


def get_enhanced_platform_analysis_settings(platform: str, video_duration: float |None = None) -> dict:
    """Get analysis settings optimized for each platform with adaptive sampling"""
    
    base_settings = {
        'tiktok': {
            'max_duration': 300,
            'base_interval': 3,  # Reduced for better coverage of quick transitions
            'max_frames': 30,
            'audio_priority': True,
            'fps_filter': 'fps=1/3',
            'scene_change_detection': True,  # Important for TikTok cuts
            'min_frames': 8  # Ensure minimum coverage
        },
        'twitter': {
            'max_duration': 140,
            'base_interval': 3,  # Reduced for better short video coverage
            'max_frames': 20,  # Increased slightly
            'audio_priority': True,
            'fps_filter': 'fps=1/3',
            'scene_change_detection': True,
            'min_frames': 6
        },
        'instagram': {
            'max_duration': 300,
            'base_interval': 5,  # Better for stories/reels
            'max_frames': 25,
            'audio_priority': True,
            'fps_filter': 'fps=1/5',
            'scene_change_detection': True,
            'min_frames': 8
        },
        'youtube': {
            'max_duration': 1800,
            'base_interval': 15,  # Reduced from 30s
            'max_frames': 15,  # Increased from 10
            'audio_priority': True,
            'fps_filter': 'fps=1/15',
            'scene_change_detection': False,  # Less critical for longer content
            'min_frames': 8
        },
        'default': {
            'max_duration': 600,
            'base_interval': 8,  # More frequent than 15s
            'max_frames': 15,
            'audio_priority': True,
            'fps_filter': 'fps=1/8',
            'scene_change_detection': True,
            'min_frames': 6
        }
    }
    
    settings = base_settings.get(platform, base_settings['default']).copy()
    
    # Adaptive adjustment based on video duration
    if video_duration:
        # For very short videos, sample more frequently
        if video_duration <= 30:
            settings['base_interval'] = min(settings['base_interval'], 2)
            settings['fps_filter'] = 'fps=1/2'
        # For very long videos, ensure we don't miss key moments
        elif video_duration > 600:
            settings['max_frames'] = min(25, int(video_duration / 30))
    
    return settings


def get_video_duration(direct_video_url: str) -> float:
    """Get video duration using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", direct_video_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            duration = float(data.get("format", {}).get("duration", 0))
            logger.info(f"Video duration: {duration:.1f} seconds")
            return duration
    except Exception as e:
        logger.warning(f"Could not get video duration: {e}")
    
    return 0


def get_advanced_frame_extraction_cmd(direct_video_url: str, frames_dir: str, settings: dict) -> list:
    """Generate advanced FFmpeg command for better frame extraction"""
    
    base_cmd = [
        "ffmpeg", "-i", direct_video_url,
        "-t", str(settings['max_duration'])
    ]
    
    # Scene change detection for platforms that need it
    if settings.get('scene_change_detection', False):
        # Combine regular interval + scene change detection
        # This will capture frames at regular intervals AND when scene changes occur
        video_filter = f"select='gte(t*{1/settings['base_interval']},n)+gt(scene,0.3)',scale=-1:720"
    else:
        # Standard interval-based extraction
        video_filter = f"fps={1/settings['base_interval']},scale=-1:720"
    
    cmd = base_cmd + [
        "-vf", video_filter,
        "-vsync", "vfr",  # Variable frame rate to handle scene detection
        "-frame_pts", "1",  # Preserve timing info
        "-threads", "4",
        "-preset", "ultrafast",
        "-q:v", "2",  # Higher quality frames for better vision model input
        f"{frames_dir}/frame_%04d.jpg"
    ]
    
    return cmd


def validate_frame_coverage(frame_files: list, video_duration: float, platform: str) -> dict:
    """Validate if frame extraction provides good coverage"""
    
    if not frame_files:
        return {"status": "failed", "reason": "No frames extracted"}
    
    settings = get_enhanced_platform_analysis_settings(platform, video_duration)
    min_frames = settings.get('min_frames', 5)
    
    coverage_ratio = len(frame_files) / video_duration if video_duration > 0 else 0
    
    if len(frame_files) < min_frames:
        return {
            "status": "insufficient", 
            "reason": f"Only {len(frame_files)} frames for {video_duration:.1f}s video",
            "recommendation": "Reduce frame interval"
        }
    
    # Platform-specific coverage validation
    if platform == 'tiktok' and coverage_ratio < 0.2:  # At least 1 frame per 5 seconds
        return {
            "status": "sparse",
            "reason": "TikTok content changes quickly, need more frames",
            "recommendation": "Increase sampling rate"
        }
    
    if platform == 'youtube' and len(frame_files) > 20:
        return {
            "status": "excessive",
            "reason": "Too many frames may overwhelm vision model",
            "recommendation": "Consider key moment detection"
        }
    
    return {
        "status": "good",
        "frames": len(frame_files),
        "coverage_ratio": coverage_ratio,
        "duration": video_duration
    }


def get_platform_optimized_vision_prompt(platform: str, frame_count: int) -> str:
    """Generate optimized prompts for vision model based on platform"""
    
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


def get_direct_video_url(url: str) -> tuple[str, str]:
    """Extract direct video URL using yt-dlp subprocess"""
    logger.info(f"Extracting direct video URL for: {url}")
    
    platform = identify_platform(url)
    logger.info(f"Detected platform: {platform}")
    
    try:
        if platform == 'direct':
            logger.info("URL is already a direct video file")
            return url, platform

        # Get platform-specific options
        cmd_options = get_platform_specific_options(platform)
        cmd_options.append(url)
        
        logger.info(f"Running yt-dlp with options: {' '.join(cmd_options[:-1])} [URL]")
        
        result = subprocess.run(
            cmd_options,
            capture_output=True, 
            text=True, 
            timeout=45  # Increased timeout for social media platforms
        )
        
        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            
            # Try with more generic options if platform-specific failed
            if platform != 'unknown':
                logger.info("Retrying with generic options...")
                generic_cmd = ["yt-dlp", "--get-url", "-f", "best", url]
                result = subprocess.run(
                    generic_cmd,
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode != 0:
                    raise HTTPException(status_code=400, detail=f"yt-dlp error: {result.stderr}")
            else:
                raise HTTPException(status_code=400, detail=f"yt-dlp error: {result.stderr}")

        direct_url = result.stdout.strip()
        logger.info(f"Extracted direct URL: {direct_url[:100]}...")
        return direct_url, platform

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timeout")
        raise HTTPException(status_code=408, detail="Timeout while extracting video URL")
    except Exception as e:
        logger.exception("Failed to extract video URL")
        raise HTTPException(status_code=400, detail=f"Failed to extract video URL: {str(e)}")


def encode_image(image_path: str) -> str:
    """Encode image to base64 for OpenAI Vision API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_frame_captions_with_openai(frames_dir: str, frame_files: list, platform: str) -> list:
    """Generate captions for video frames using OpenAI Vision API in a single call"""
    frame_summaries = []
    
    # Get platform-optimized prompt
    platform_prompt = get_platform_optimized_vision_prompt(platform, len(frame_files))

    # Prepare content list with proper typing
    content: list[Any] = [
        {
            "type": "text",
            "text": platform_prompt
        }
    ]
    
    # Add all images to the content list
    for idx, fname in enumerate(frame_files, start=1):
        frame_path = os.path.join(frames_dir, fname)
        try:
            base64_image = encode_image(frame_path)
            content.extend([
                {
                    "type": "text",
                    "text": f"\nFrame {idx}:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"  # Use low detail for faster processing and lower costs
                    }
                }
            ])
        except Exception as img_error:
            logger.error(f"Error reading {fname}: {img_error}")
            # Skip this frame but continue with others
            continue

    # Prepare messages with proper structure
    messages: list[Any] = [
        {
            "role": "user",
            "content": content
        }
    ]

    # Send all frames in a single API call
    try:
        logger.info(f"Sending all {len(frame_files)} frames to OpenAI Vision API in single call...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency with vision
            messages=messages,
            max_tokens=2000,  # Increased for multiple frames
            temperature=0.3
        )
        
        captions_text = response.choices[0].message.content
        
        # Parse the response and extract frame captions
        if captions_text:
            lines = captions_text.splitlines()
            for line in lines:
                line = line.strip()
                if line and ("frame" in line.lower() or any(char.isdigit() for char in line[:10])):
                    frame_summaries.append(line)
        
        logger.info(f"Generated {len(frame_summaries)} frame captions in single API call")
        
        # Ensure we have captions for all frames (fallback if parsing failed)
        if len(frame_summaries) < len(frame_files):
            logger.warning(f"Only got {len(frame_summaries)} captions for {len(frame_files)} frames")
            # Add generic captions for missing frames
            for i in range(len(frame_summaries), len(frame_files)):
                frame_summaries.append(f"Frame {i+1}: Video content frame")
        
    except Exception as caption_error:
        logger.error(f"Error generating captions: {caption_error}")
        # Fallback: create basic frame descriptions
        for idx in range(1, len(frame_files) + 1):
            frame_summaries.append(f"Frame {idx}: Caption generation failed - {platform} video content")

    return frame_summaries


@app.get("/")
def home():
    """Serve the main HTML page"""
    logger.info("Serving homepage")
    return FileResponse("static/index.html")


@app.post("/summarize_video")
def summarize_video(req: VideoRequest):
    job_dir, direct_video_url, platform = None, None, None
    logger.info(f"Received video summarization request for {req.video_url}")
    
    try:
        direct_video_url, platform = get_direct_video_url(req.video_url)
        
        # Get video duration for adaptive settings
        video_duration = get_video_duration(direct_video_url)
        
        # Get enhanced analysis settings
        analysis_settings = get_enhanced_platform_analysis_settings(platform, video_duration)
        
        logger.info(f"Using enhanced analysis settings for {platform}: {analysis_settings}")

        # Setup working directories
        job_dir = os.path.join(WORK_DIR, "job")
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        os.makedirs(job_dir, exist_ok=True)
        audio_path = f"{job_dir}/audio.mp3"
        frames_dir = f"{job_dir}/frames"
        os.makedirs(frames_dir, exist_ok=True)

        # 1. Extract audio using platform-optimized settings
        transcript = ""
        logger.info("Extracting audio with platform-optimized FFmpeg settings...")
        
        try:
            # Use platform-specific duration limit
            audio_cmd = [
                "ffmpeg", "-i", direct_video_url, 
                "-t", str(analysis_settings['max_duration']),
                "-q:a", "0",   # High quality audio
                "-map", "a",   # Map only audio stream
                "-ac", "1",    # Convert to mono
                "-ar", "16000", # 16kHz sample rate
                audio_path, "-y"
            ]
            
            result = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg audio extraction failed: {result.stderr}")
                logger.info("Proceeding with visual-only analysis")
            else:
                logger.info("Audio extracted successfully with platform-optimized settings")
                
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio extraction timed out")
            logger.info("Proceeding with visual-only analysis")
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            logger.info("Proceeding with visual-only analysis")

        # 2. Transcribe with platform awareness
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
            try:
                logger.info(f"Transcribing {platform} audio with Whisper...")
                model = whisper.load_model("base")
                whisper_result = model.transcribe(audio_path)
                transcript = whisper_result["text"].strip() # type: ignore
                
                if transcript:
                    logger.info(f"Transcript generated successfully ({len(transcript)} characters)")
                else:
                    logger.info(f"Transcript is empty - likely music/{platform} video without speech")
                    transcript = ""
                    
            except Exception as whisper_error:
                logger.warning(f"Whisper transcription failed: {whisper_error}")
                transcript = ""

        # 3. Extract frames with enhanced settings
        logger.info(f"Extracting frames with enhanced settings for {platform}...")
        try:
            frame_cmd = get_advanced_frame_extraction_cmd(direct_video_url, frames_dir, analysis_settings)
            
            result = subprocess.run(frame_cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg frame extraction failed: {result.stderr}")
                # Fallback to simple extraction
                logger.info("Attempting fallback frame extraction...")
                fallback_cmd = [
                    "ffmpeg", "-i", direct_video_url, 
                    "-t", str(min(analysis_settings['max_duration'], 300)),
                    "-vf", f"fps={1/analysis_settings['base_interval']},scale=-1:720",
                    "-threads", "4",
                    "-preset", "ultrafast",
                    "-q:v", "3",
                    f"{frames_dir}/frame_%04d.jpg"
                ]
                result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    raise HTTPException(status_code=400, detail="Failed to extract frames")
                
        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            raise HTTPException(status_code=408, detail="Frame extraction timed out")
            
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        # Limit frames based on platform settings
        frame_files = frame_files[:analysis_settings['max_frames']]
        
        # Validate frame coverage
        coverage_info = validate_frame_coverage(frame_files, video_duration, platform)
        logger.info(f"Frame coverage validation: {coverage_info}")
        
        logger.info(f"Extracted {len(frame_files)} frames for enhanced {platform} analysis")

        # 4. Generate captions with enhanced OpenAI Vision
        logger.info("Generating enhanced frame captions with OpenAI Vision API...")
        frame_summaries = generate_frame_captions_with_openai(frames_dir, frame_files, platform)
        logger.info(f"Generated {len(frame_summaries)} enhanced frame captions")

        # 5. Generate platform-aware summary using Gemini
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

TRANSCRIPT:
{transcript[:2000]}

VISUAL SNAPSHOTS (enhanced frame analysis with {len(frame_summaries)} frames):
{chr(10).join(frame_summaries)}

TASK:
Provide a structured, detailed summary that combines both the audio and visual elements. 

1. Identify what the video is about (e.g., gaming → name the game, anime → name the anime, educational → name the topic, music → name the song/artist, Dance → name the dance).
2. Highlight the **key points or moments using the transcribed audio if its available and understandable** (e.g., timestamps, scenes, or sections of interest).
3. Explain **why the video is engaging** for {platform.upper()} users (trends, entertainment, learning value, cultural relevance, etc.).
4. Write in a **clear, audience-friendly way** (easy to read, short paragraphs, avoid jargon).
"""  
        else:
            combined_context = f"""
{platform_summary_context.get(platform, platform_summary_context['default'])}

VISUAL SNAPSHOTS (enhanced frame analysis with {len(frame_summaries)} frames):
{chr(10).join(frame_summaries)}

TASK:
Provide a structured, detailed summary of this video using only the enhanced visual analysis.

1. Identify the type of content (music, dance, anime, gaming, tutorial, lifestyle, etc.).
2. Mention any recognizable people, characters, brands(if not famous, classify as upcoming/independent).
3. Highlight visual trends, styles, and effects that make it engaging for {platform.upper()} users.
4. Explain the likely **audience appeal** (why someone would watch/share it).
"""
        logger.info(f"Requesting enhanced {platform}-aware summary from Gemini...")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')  # type: ignore
            response = model.generate_content(combined_context)
            summary = response.text
            logger.info(f"Enhanced platform-aware summary generated successfully for {platform}")
            
        except Exception as e:
            logger.error(f"Gemini summary generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

        # Cleanup
        shutil.rmtree(job_dir)
        logger.info("Cleaned up job directory")

        return {
            "summary": summary,
            "platform": platform,
            "transcript_excerpt": transcript[:500] if transcript else f"No transcript available (music/{platform} video)",
            "frames_analyzed": len(frame_files),
            "frames_used": frame_files,
            "original_url": req.video_url,
            "processing_url": direct_video_url[:100] + "...",
            "has_audio_transcript": bool(transcript.strip()),
            "video_duration": video_duration,
            "frame_coverage_info": coverage_info,
            "analysis_settings_used": analysis_settings,
            "vision_api_used": "OpenAI GPT-4 Vision (Enhanced)",
            "enhancements": [
                "Adaptive frame sampling",
                "Scene change detection",
                "Platform-optimized prompts",
                "Enhanced frame quality",
                "Coverage validation"
            ]
        }

    except Exception as e:
        logger.exception("Unexpected error during summarization")
        if job_dir and os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        raise


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "supported_platforms": ["TikTok", "Twitter/X", "Instagram", "YouTube", "Direct URLs"],
        "yt_dlp_available": shutil.which("yt-dlp") is not None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
        "vision_api": "OpenAI GPT-4 Vision (Enhanced)",
        "summary_api": "Google Gemini",
        "enhancements": [
            "Adaptive frame sampling based on video duration",
            "Scene change detection for dynamic content",
            "Platform-specific analysis settings",
            "Enhanced frame quality and coverage validation",
            "Platform-optimized vision prompts"
        ]
    }


if __name__ == "__main__":
    logger.info("Starting enhanced FastAPI server with improved visual analysis...")
    uvicorn.run(app, host="0.0.0.0", port=8000)