# Enhanced Video Analysis Web Application - Integrated Main Module
# This version integrates Vector Store, Video Pipeline, Audio Pipeline, and System modules

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
import whisper
import google.generativeai as genai
from typing import Any, List, Optional, Dict, Tuple
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from pathlib import Path
import cv2
import torch
from PIL import Image
import open_clip
import ffmpeg
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.staticfiles import StaticFiles
import requests
import vector_store
import json
import hashlib

# Import standalone modules
from audio_pipeline import AudioPipeline as StandaloneAudioPipeline
from timeline import MasterTimeline, AudioEvent, FrameEvent
from video_pipeline import VideoPipeline

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

# ---------------- Global Model Loading ----------------
# Global variables for pre-loaded models
global_whisper_model = None
global_clip_model = None
global_clip_preprocess = None
global_sentence_transformer = None
global_device = None

def load_models_at_startup():
    """Load all models at startup to avoid on-demand loading delays"""
    global global_whisper_model, global_clip_model, global_clip_preprocess, global_sentence_transformer, global_device
    
    logger.info("Starting global model loading...")
    
    # Set device
    global_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {global_device}")
    
    # Load Whisper model
    try:
        logger.info("Loading Whisper model...")
        global_whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise
    
    # Load CLIP model
    try:
        logger.info("Loading CLIP model...")
        global_clip_model, _, global_clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            device=global_device
        )
        global_clip_model.eval()
        logger.info("CLIP model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise
    
    # Load Sentence Transformer
    try:
        logger.info("Loading Sentence Transformer model...")
        global_sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence Transformer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Sentence Transformer: {e}")
        raise
    
    logger.info("All models loaded successfully!")

# Load models at startup
load_models_at_startup()

# ---------------- Configuration ----------------
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

SUMMARY_CACHE_FILE = "summary_cache.json"


# ---------------- Caching Functions ----------------
def load_summary_cache() -> Dict[str, Any]:
    """Load the summary cache from a file"""
    if os.path.exists(SUMMARY_CACHE_FILE):
        try:
            with open(SUMMARY_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load summary cache: {e}")
    return {}

def save_summary_cache(cache: Dict[str, Any]) -> None:
    """Save the summary cache to a file"""
    try:
        with open(SUMMARY_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except IOError as e:
        logger.error(f"Could not save summary cache: {e}")

# Load cache at startup
summary_cache = load_summary_cache()

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# ---------------- Data Models ----------------
class VideoRequest(BaseModel):
    video_url: str

# ---------------- Vector Store Module ----------------
class VectorStore:
    """Base class for vector storage and retrieval"""
    def __init__(self, dimension: int, store_type: str = "faiss"):
        self.dimension = dimension
        self.store_type = store_type
        self.index = None
        self.metadata = []
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        raise NotImplementedError
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        raise NotImplementedError

class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for fast similarity search"""
    def __init__(self, dimension: int):
        super().__init__(dimension, "faiss")
        
        # Create FAISS index (using L2 distance with normalization = cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors
        
        # Use GPU if available
        if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS")
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
        
        self.metadata = []
        logger.info(f"Initialized FAISS index with dimension {dimension}")
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings to the index with proper error handling"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("Number of metadata entries must match number of embeddings")
        
        # Ensure embeddings are 2D and float32
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Search for similar embeddings"""
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Get metadata for results
        result_metadata = [self.metadata[idx] for idx in indices[0]]
        
        return distances[0], indices[0], result_metadata
    
    
# ---------------- Audio Pipeline Module ----------------
class EnhancedAudioPipeline(StandaloneAudioPipeline):
    """Enhanced Audio Pipeline that uses pre-loaded global models"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        # Pass the pre-loaded global models directly
        super().__init__(
            whisper_model=global_whisper_model,
            embed_model=global_sentence_transformer,
            temp_dir=temp_dir
        )
        
        logger.info("EnhancedAudioPipeline initialized with pre-loaded models")
    
    def get_audio_embeddings(self, transcript: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate embeddings for transcribed audio segments"""
        texts = [segment['text'] for segment in transcript]
        
        if not texts:
            return np.array([]), []
            
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
        
        # Create metadata for each embedding
        metadata = [
            {"type": "audio", "timestamp": segment['start'], "text": segment['text']}
            for segment in transcript
        ]
        
        return embeddings, metadata
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding using Sentence Transformer"""
        return self.embed_model.encode(text, convert_to_numpy=True)
    
    def transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe audio using the parent class method"""
        # Use the parent transcribe_audio method directly
        return super().transcribe_audio(audio_path)
    
    def create_audio_events(self, transcription: Dict[str, Any]) -> List[AudioEvent]:
        """Create AudioEvent objects from transcription segments"""
        audio_events = []
        
        for segment in transcription["segments"]:
            event = AudioEvent(
                timestamp=segment["start"],
                text=segment["text"].strip(),
                metadata={
                    'start': segment["start"],
                    'end': segment["end"]
                }
            )
            audio_events.append(event)
        
        logger.info(f"Created {len(audio_events)} audio events")
        return audio_events
    
    def generate_embeddings(self, audio_events: List[AudioEvent]) -> np.ndarray:
        """Generate embeddings for audio transcript segments"""
        # Use the parent class method directly
        return super().generate_embeddings(audio_events)
    
# ---------------- System Integration ----------------
class VideoAnalyticsSystem:
    """Complete Video Analytics System that orchestrates all pipeline components"""
    def __init__(self):
        self.vector_store = None
        self.audio_pipeline = None
        self.video_pipeline = None
        self.timeline: Optional[MasterTimeline] = None
        
        logger.info("Initialized VideoAnalyticsSystem")
    
    def initialize_components(self):
        """Initialize pipeline components"""
        self.vector_store = vector_store.MultimodalVectorStore()
        self.audio_pipeline = EnhancedAudioPipeline()
        self.video_pipeline = VideoPipeline()
        logger.info("All components initialized")
    
    def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """Process a complete video through the pipeline with intelligent frame-transcript mixing"""
        logger.info(f"=== Processing Video: {video_path} ===")
        
        # Initialize components if not already done
        if self.vector_store is None:
            self.initialize_components()
        
        video_path_obj = Path(video_path)
        
        # Get video duration first
        try:
            probe = ffmpeg.probe(str(video_path_obj))
            video_duration = float(probe['format'].get('duration', 0))
            logger.info(f"Video duration: {video_duration} seconds")
        except Exception as e:
            logger.warning(f"Could not get video duration: {str(e)}")
            video_duration = 0
        
        # Step 1: Audio Pipeline
        logger.info("[1/3] Audio Pipeline")
        audio_path = self.audio_pipeline.extract_audio(video_path_obj)
        transcription = self.audio_pipeline.transcribe_audio(audio_path)
        audio_events = self.audio_pipeline.create_audio_events(transcription)
        audio_embeddings = self.audio_pipeline.generate_embeddings(audio_events)
        
        # Add audio embeddings to vector store
        audio_metadata = [event.to_dict() for event in audio_events]
        self.vector_store.add("audio", audio_embeddings, audio_metadata)
        
        logger.info(f"Audio events: {len(audio_events)}")
        logger.info(f"Audio embeddings shape: {audio_embeddings.shape}")
        
        # Step 2: Video Pipeline (Frame Extraction)
        logger.info("[2/3] Video Pipeline - Frame Extraction")
        frames = self.video_pipeline.extract_frames(video_path_obj, video_id, save_frames=True)
        frame_events = self.video_pipeline.create_frame_events(frames, video_id)
        frame_embeddings = self.video_pipeline.generate_embeddings(frames)
        
        logger.info(f"Frame events: {len(frame_events)}")
        logger.info(f"Frame embeddings shape: {frame_embeddings.shape}")
        
        # Step 3: Intelligent Frame-Transcript Mixing (NEW!)
        logger.info("[3/3] Intelligent Frame-Transcript Mixing")
        
        # Match frames to transcript segments based on timestamps
        enhanced_frame_events = self._create_intelligent_frame_descriptions(
            frames, frame_events, audio_events
        )
        
        # Initialize timeline with proper duration
        self.timeline = MasterTimeline(video_id=video_id, duration_seconds=video_duration, metadata={})
        
        # Add audio events to timeline
        for event in audio_events:
            self.timeline.add_audio_event(event)
        
        # Add enhanced frame events to timeline (with descriptions!)
        for event in enhanced_frame_events:
            self.timeline.add_frame_event(event)
        
        # Add enhanced video embeddings to vector store
        frame_metadata = [event.to_dict() for event in enhanced_frame_events]
        self.vector_store.add("video", frame_embeddings, frame_metadata)
        
        logger.info("=== Video Processing Complete ===")
        
        return {
            "video_id": video_id,
            "audio_events": len(audio_events),
            "frame_events": len(enhanced_frame_events),
            "audio_embeddings_shape": audio_embeddings.shape,
            "frame_embeddings_shape": frame_embeddings.shape,
            "visual_segments_created": len(enhanced_frame_events)
        }

    def _create_intelligent_frame_descriptions(self, frames: List[Tuple], frame_events: List[FrameEvent], 
                                            audio_events: List[AudioEvent]) -> List[FrameEvent]:
        """Create intelligent frame descriptions by matching with transcript segments"""
        
        enhanced_events = []
        
        # Create fixed 10-second segments covering the entire video duration
        segment_duration = 10.0
        
        # Find the maximum timestamp to determine total video duration
        if frames:
            max_timestamp = max(timestamp for timestamp, _, _ in frames)
            total_segments = int(max_timestamp // segment_duration) + 1
        else:
            total_segments = 1
        
        # Process each fixed segment
        for segment_index in range(total_segments):
            segment_start = segment_index * segment_duration
            segment_end = (segment_index + 1) * segment_duration
            
            # Find frames that belong to this segment
            segment_frames = []
            segment_events = []
            
            for (timestamp, frame_array, frame_path), frame_event in zip(frames, frame_events):
                if segment_start <= timestamp < segment_end:
                    segment_frames.append((timestamp, frame_array, frame_path))
                    segment_events.append(frame_event)
            
            # Process this segment if it has frames
            if segment_frames:
                enhanced_segment_events = self._analyze_time_segment(
                    segment_start, segment_end,
                    segment_frames, segment_events, audio_events
                )
                enhanced_events.extend(enhanced_segment_events)
        return enhanced_events

    def _analyze_time_segment(self, segment_start: float, segment_end: float, 
                            frames: List[Tuple], frame_events: List[FrameEvent], 
                            audio_events: List[AudioEvent]) -> List[FrameEvent]:
        """Analyze a time segment and create intelligent descriptions"""
        
        # Get transcript content for this time segment
        segment_transcript = []
        for audio_event in audio_events:
            if (audio_event.timestamp >= segment_start - 2.0 and 
                audio_event.timestamp <= segment_end + 2.0 and 
                audio_event.text):
                segment_transcript.append({
                    'timestamp': audio_event.timestamp,
                    'text': audio_event.text
                })
        
        # Sort transcript by timestamp
        segment_transcript.sort(key=lambda x: x['timestamp'])
        
        # Create varied descriptions for each frame in this segment
        for i, (frame_event, (timestamp, frame_array, frame_path)) in enumerate(zip(frame_events, frames)):
            # Calculate frame position within segment (0 to 1)
            segment_progress = (timestamp - segment_start) / (segment_end - segment_start) if segment_end != segment_start else 0
            
            # Find transcript content based on frame position and timestamp
            frame_description = None
            closest_transcript = None
            
            if segment_transcript:
                # Strategy 1: Find transcript closest to this frame's timestamp
                closest_distance = float('inf')
                for transcript_item in segment_transcript:
                    distance = abs(transcript_item['timestamp'] - timestamp)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_transcript = transcript_item
                
                # Strategy 2: Distribute transcript content across frames based on position
                transcript_index = int(segment_progress * len(segment_transcript))
                transcript_index = min(transcript_index, len(segment_transcript) - 1)
                
                # Use closest transcript but vary the description based on frame position
                if closest_transcript:
                    # Create different description styles based on frame position
                    if segment_progress < 0.3:
                        # Early frames: focus on beginning of transcript
                        frame_description = f"Early segment [{timestamp:.1f}s]: {closest_transcript['text'][:80]}..."
                    elif segment_progress < 0.7:
                        # Middle frames: show current context
                        frame_description = f"Mid segment [{timestamp:.1f}s]: {closest_transcript['text'][:80]}..."
                    else:
                        # Later frames: show progression
                        frame_description = f"Late segment [{timestamp:.1f}s]: {closest_transcript['text'][:80]}..."
                    
                    # Add transcript timestamp for reference
                    frame_description += f" (audio at {closest_transcript['timestamp']:.1f}s)"
                    
                    # For longer transcripts, show different parts based on frame position
                    if len(closest_transcript['text']) > 80:
                        start_pos = int(segment_progress * (len(closest_transcript['text']) - 80))
                        excerpt = closest_transcript['text'][start_pos:start_pos + 80]
                        frame_description = f"Segment {segment_start:.1f}s-{segment_end:.1f}s [{timestamp:.1f}s]: {excerpt}..."
                else:
                    # Fallback to segment summary
                    transcript_summary = " ".join([item['text'] for item in segment_transcript[:2]])
                    frame_description = f"Segment {segment_start:.1f}s-{segment_end:.1f}s [{timestamp:.1f}s]: {transcript_summary[:80]}..."
            else:
                frame_description = f"Visual content at {timestamp:.1f}s (no matching audio)"
            
            # Enhance this specific frame event
            frame_event.metadata['description'] = frame_description
            frame_event.metadata['segment_start'] = segment_start
            frame_event.metadata['segment_end'] = segment_end
            frame_event.metadata['frame_timestamp'] = timestamp
            frame_event.metadata['closest_transcript'] = closest_transcript['text'] if closest_transcript else None
            frame_event.metadata['segment_progress'] = segment_progress
        
        return frame_events
    def _match_frames_to_transcript(self, frames: List[Tuple], audio_events: List[AudioEvent]) -> List[Dict]:
        """Match frames to their corresponding transcript segments based on timestamp"""
        matches = []
        
        for timestamp, frame_array, frame_path in frames:
            # Find the audio event that corresponds to this timestamp
            matching_audio = None
            for audio_event in audio_events:
                if abs(audio_event.timestamp - timestamp) <= 2.0:  # 2-second tolerance
                    matching_audio = audio_event
                    break
            
            matches.append({
                'timestamp': timestamp,
                'frame_array': frame_array,
                'frame_path': frame_path,
                'audio_event': matching_audio
            })
    
        return matches
    
    def _create_frame_batches(self, frame_matches: List[Dict], max_batch_size: int = 8) -> List[Dict]:
        """Group frames into intelligent batches for AI analysis"""
        batches = []
        current_batch = []
        current_time_range = {'start': None, 'end': None}
        
        for match in frame_matches:
            if not current_batch:
                current_time_range['start'] = match['timestamp']
            
            current_batch.append(match)
            current_time_range['end'] = match['timestamp']
            
            if len(current_batch) >= max_batch_size:
                batches.append({
                    'frames': current_batch,
                    'time_range': current_time_range.copy(),
                    'frame_count': len(current_batch)
                })
                current_batch = []
                current_time_range = {'start': None, 'end': None}
        
        # Add remaining frames
        if current_batch:
            batches.append({
                'frames': current_batch,
                'time_range': current_time_range,
                'frame_count': len(current_batch)
            })
        
        return batches
    
    def _get_batch_transcript_context(self, batch: Dict, audio_events: List[AudioEvent]) -> str:
        """Extract transcript context for a batch of frames"""
        context_parts = []
        batch_start = batch['time_range']['start']
        batch_end = batch['time_range']['end']
        
        for audio_event in audio_events:
            if (audio_event.timestamp >= batch_start - 5.0 and 
                audio_event.timestamp <= batch_end + 5.0 and 
                audio_event.text):
                context_parts.append(f"[{audio_event.timestamp:.1f}s] {audio_event.text}")
        
        return " ".join(context_parts) if context_parts else "No transcript available"
    
    def _generate_batch_description(self, frames: List[Dict], transcript_context: str, time_range: Dict) -> str:
        """Generate AI description for a batch of frames using Gemini"""
        try:
            # Create a prompt that includes transcript context
            prompt = f"""Analyze these video frames and provide a detailed visual description.

    Transcript context for this time period ({time_range['start']:.1f}s - {time_range['end']:.1f}s):
    {transcript_context}

    Please describe:
    1. What is visually happening in these frames
    2. Key objects, people, or actions visible
    3. How the visuals relate to the transcript context
    4. Any important visual details that complement the audio

    Provide a comprehensive visual description that will help someone understand what they would see in this part of the video."""

            # For now, return a placeholder description
            # You should integrate with your Gemini API here
            return f"Visual analysis of frames from {time_range['start']:.1f}s to {time_range['end']:.1f}s. {transcript_context}"
            
        except Exception as e:
            logger.error(f"Error generating batch description: {str(e)}")
            return f"Visual content from {time_range['start']:.1f}s to {time_range['end']:.1f}s"
    
    def search_video(self, query: str, query_type: str = "both", top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for a query in the video content"""
        results = []
        
        if query_type in ["audio", "both"]:
            # Search in audio
            logger.info(f"Searching audio for: '{query}'")
            audio_embedding = self.audio_pipeline.embed_text(query)
            search_results = self.vector_store.search("audio", audio_embedding, top_k)
            
            for result in search_results:
                results.append({
                    "modality": "audio",
                    "score": result.get("score", 0),
                    "timestamp": result.get("timestamp"),
                    "text": result.get("text")
                })
        
        if query_type in ["video", "both"]:
            # Search in video frames
            logger.info(f"Searching video for: '{query}'")
            frame_embedding = self.video_pipeline.embed_text(query)
            search_results = self.vector_store.search("video", frame_embedding, top_k)
            
            for result in search_results:
                results.append({
                    "modality": "video",
                    "score": result.get("score", 0),
                    "timestamp": result.get("timestamp"),
                    "preview_url": result.get("preview_url")
                })
        
        if not results:
            logger.info(f"No results found for query: '{query}'")
            return []

        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    def get_video_content(self, video_id: str) -> dict:
        """Extract actual content from video processing results with dynamic visual description generation"""
        try:
            # Initialize content structures
            transcript = []
            visual_descriptions = []
            
            # Check if we have a timeline with content
            if self.timeline and self.timeline.video_id == video_id:
                # Extract transcript from audio events
                for event in self.timeline.audio_events:
                    if hasattr(event, 'text') and event.text:
                        transcript.append({
                            "timestamp": event.timestamp,
                            "text": event.text
                        })
                
                # Generate visual descriptions dynamically from frame events
                if self.timeline.frame_events:
                    # Group frames into segments for description generation
                    segment_size = max(1, len(self.timeline.frame_events) // 10)  # Create ~10 segments
                    
                    for i in range(0, len(self.timeline.frame_events), segment_size):
                        segment_frames = self.timeline.frame_events[i:i + segment_size]
                        
                        if segment_frames:
                            # Get first and last frame timestamps for the segment
                            start_time = segment_frames[0].timestamp
                            end_time = segment_frames[-1].timestamp
                            
                            # Prepare frame data for description generation
                            frame_dicts = []
                            for frame_event in segment_frames:
                                if frame_event.frame_path and os.path.exists(frame_event.frame_path):
                                    frame_dicts.append({
                                        'path': frame_event.frame_path,
                                        'timestamp': frame_event.timestamp
                                    })
                            
                            # Generate visual description for this segment
                            if frame_dicts:
                                try:
                                    # Use the same method as query analyzer for consistency
                                    visual_description = self._generate_visual_description_for_segment(frame_dicts)
                                    
                                    visual_descriptions.append({
                                        "timestamp": start_time,
                                        "description": visual_description
                                    })
                                except Exception as e:
                                    logger.warning(f"Could not generate visual description for segment {start_time}s-{end_time}s: {e}")
                                    # Fallback to basic description
                                    visual_descriptions.append({
                                        "timestamp": start_time,
                                        "description": f"Video segment from {start_time:.1f}s to {end_time:.1f}s with {len(segment_frames)} frames"
                                    })
            
            # Fallback: Try to get content from vector store metadata
            if not transcript and self.vector_store:
                try:
                    # Search for audio content in vector store
                    dummy_query = np.zeros((1, 384))  # Dummy embedding for search
                    audio_results = self.vector_store.search("audio", dummy_query, k=100)
                    
                    for result in audio_results:
                        if result.get('text'):
                            transcript.append({
                                "timestamp": result.get('timestamp', 0),
                                "text": result['text']
                            })
                except Exception as e:
                    logger.warning(f"Could not extract audio content from vector store: {e}")
            
            return {
                "transcript": transcript,
                "visual_descriptions": visual_descriptions,
                "total_audio_segments": len(transcript),
                "total_visual_segments": len(visual_descriptions)
            }
            
        except Exception as e:
            logger.error(f"Error extracting video content: {str(e)}")
            return {
                "transcript": [],
                "visual_descriptions": [],
                "total_audio_segments": 0,
                "total_visual_segments": 0
            }
            
    def _generate_visual_description_for_segment(self, frame_dicts: List[Dict]) -> str:
        """Generate visual description for a segment using Gemini vision model"""
        try:
            # Import required modules
            from PIL import Image
            import google.generativeai as genai
            
            # Prepare the prompt for the vision model
            prompt = "Describe the key visual elements and actions across these frames in a single, concise sentence."
            
            # Select a subset of frames to pass to the model (limit to avoid overwhelming the API)
            max_frames_to_llm = min(len(frame_dicts), 5)  # Limit to 5 frames max
            indices = np.linspace(0, len(frame_dicts) - 1, num=max_frames_to_llm, dtype=int)
            selected_frames = [frame_dicts[i] for i in indices]
            
            # Prepare the images for the model
            model_input: List[Any] = [prompt]
            for frame in selected_frames:
                try:
                    img = Image.open(frame['path'])
                    model_input.append(img)
                except FileNotFoundError:
                    logger.warning(f"Frame not found at {frame['path']}, skipping.")
                    continue
            
            if len(model_input) <= 1:
                return "Could not load frames for description."
            
            # Call the Gemini API
            try:
                model = genai.GenerativeModel('gemini-2.5-flash-lite')
                response = model.generate_content(model_input)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Error generating visual description with Gemini: {e}")
                return "Error generating visual description."
                
        except Exception as e:
            logger.error(f"Error in _generate_visual_description_for_segment: {e}")
            return "Visual description unavailable."
    def is_video_processed(self, video_id: str) -> bool:
        """Check if a video has already been processed"""
        if self.timeline is None:
            return False
        
        # Check if video_id matches the current timeline
        if self.timeline.video_id != video_id:
            return False
        
        # Check if there's any content in the timeline
        has_audio_content = len(self.timeline.audio_events) > 0
        has_frame_content = len(self.timeline.frame_events) > 0
        
        return has_audio_content or has_frame_content
    
        
    def generate_intelligent_summary(self, video_id: str) -> str:
        """Generate intelligent summary using Gemini API with embedding-based content selection"""
        try:
            # Check if vector store is initialized
            if self.vector_store is None:
                return self.generate_basic_summary(video_id)
            
            # Search for representative content using embedding similarity
            key_transcript_items = []
            key_frame_descriptions = []
            
            # Get representative transcript content
            try:
                # Search audio store for diverse content
                if hasattr(self.vector_store, 'audio_store') and self.vector_store.audio_store.index.ntotal > 0:
                    # Create a query that should match general content
                    general_query = "main topic important content"
                    audio_query_embedding = self.audio_pipeline.embed_text(general_query)
                    audio_results = self.vector_store.search("audio", audio_query_embedding, k=5)
                    
                    for result in audio_results:
                        if 'text' in result and 'timestamp' in result:
                            key_transcript_items.append({
                                "timestamp": result.get('timestamp', 0),
                                "text": result['text']
                            })
            except Exception as e:
                logger.warning(f"Could not get representative transcript content: {e}")
            
            # Get representative frame content  
            try:
                # Search video store for diverse content
                if hasattr(self.vector_store, 'video_store') and self.vector_store.video_store.index.ntotal > 0:
                    # Create a query for visual content
                    visual_query = "key visual scene important moment"
                    video_query_embedding = self.video_pipeline.embed_text(visual_query)
                    video_results = self.vector_store.search("video", video_query_embedding, k=5)
                    
                    for result in video_results:
                        if 'description' in result and 'timestamp' in result:
                            key_frame_descriptions.append({
                                "timestamp": result.get('timestamp', 0),
                                "description": result['description']
                            })
            except Exception as e:
                logger.warning(f"Could not get representative frame content: {e}")
            
            # Fallback to basic content if embedding search fails
            if not key_transcript_items and not key_frame_descriptions:
                return self.generate_basic_summary(video_id)
            
            # Prepare content for Gemini using representative samples
            transcript_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['text']}" for item in key_transcript_items])
            visual_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['description']}" for item in key_frame_descriptions])
            
            # Create comprehensive prompt for Gemini
            prompt = f"""Please provide a comprehensive summary of this video based on the following representative content:

    AUDIO TRANSCRIPT (Key Segments):
    {transcript_text}

    VISUAL DESCRIPTIONS (Key Moments):
    {visual_text}

    Based on these representative segments, please create a detailed summary that includes:
    1. Main topics and key points discussed
    2. Important visual elements and scenes
    3. Overall narrative or message
    4. Key timestamps for important moments

    Make the summary engaging and informative for someone who hasn't watched the video."""

            # Call Gemini API
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating intelligent summary with embedding selection: {str(e)}")
            return self.generate_basic_summary(video_id)
    def get_comprehensive_video_analysis(self, video_id: str) -> Dict[str, Any]:
        """Get comprehensive video analysis with full summary, key transcript moments, and key frames with timestamps"""
        try:
            # Check if vector store is initialized
            if self.vector_store is None:
                logger.warning("Vector store not initialized, falling back to basic content")
                content = self.get_video_content(video_id)
                return {
                    "full_summary": self.generate_basic_summary(video_id),
                    "key_transcript_moments": content.get("transcript", [])[:5],
                    "key_frame_moments": content.get("visual_descriptions", [])[:5],
                    "method": "basic"
                }
            
            # Initialize results
            key_transcript_moments = []
            key_frame_moments = []
            
            # Get representative transcript content using diverse queries
            try:
                if hasattr(self.vector_store, 'audio_store') and self.vector_store.audio_store.index.ntotal > 0:
                    # Use multiple diverse queries to find key content
                    queries = [
                        "main topic important speech",
                        "key discussion points",
                        "significant dialogue",
                        "important announcement",
                        "critical information"
                    ]
                    
                    seen_texts = set()
                    for query in queries:
                        if len(key_transcript_moments) >= 5:
                            break
                            
                        audio_query_embedding = self.audio_pipeline.embed_text(query)
                        audio_results = self.vector_store.search("audio", audio_query_embedding, k=3)
                        
                        for result in audio_results:
                            if 'text' in result and 'timestamp' in result:
                                text = result['text']
                                if text not in seen_texts and len(text) > 20:  # Avoid duplicates and very short texts
                                    seen_texts.add(text)
                                    key_transcript_moments.append({
                                        "timestamp": result.get('timestamp', 0),
                                        "text": text,
                                        "duration": result.get('duration', 0),
                                        "confidence": result.get('confidence', 0)
                                    })
            except Exception as e:
                logger.warning(f"Could not get representative transcript content: {e}")
            
            # Get representative frame content using diverse queries
            try:
                if hasattr(self.vector_store, 'video_store') and self.vector_store.video_store.index.ntotal > 0:
                    # Use multiple diverse queries to find key visual moments
                    visual_queries = [
                        "key visual scene important moment",
                        "significant visual event",
                        "important visual detail",
                        "critical visual information",
                        "notable visual element"
                    ]
                    
                    seen_descriptions = set()
                    for query in visual_queries:
                        if len(key_frame_moments) >= 5:
                            break
                            
                        video_query_embedding = self.video_pipeline.embed_text(query)
                        video_results = self.vector_store.search("video", video_query_embedding, k=3)
                        
                        for result in video_results:
                            if 'description' in result and 'timestamp' in result:
                                description = result['description']
                                if description not in seen_descriptions and len(description) > 15:  # Avoid duplicates and short descriptions
                                    seen_descriptions.add(description)
                                    key_frame_moments.append({
                                        "timestamp": result.get('timestamp', 0),
                                        "description": description,
                                        "frame_index": result.get('frame_index', 0),
                                        "confidence": result.get('confidence', 0)
                                    })
            except Exception as e:
                logger.warning(f"Could not get representative frame content: {e}")
            
            # Fallback to basic content if embedding search fails
            if not key_transcript_moments and not key_frame_moments:
                content = self.get_video_content(video_id)
                key_transcript_moments = content.get("transcript", [])[:5]
                key_frame_moments = content.get("visual_descriptions", [])[:5]
            
            # Generate comprehensive summary using the key content
            transcript_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['text']}" for item in key_transcript_moments])
            visual_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['description']}" for item in key_frame_moments])
            
            # Create comprehensive prompt for Gemini
            prompt = f"""Please provide a comprehensive summary of this video based on the following key content segments:

KEY AUDIO TRANSCRIPT MOMENTS:
{transcript_text}

KEY VISUAL FRAME MOMENTS:
{visual_text}

Based on these key moments, please create a detailed summary that includes:
1. Main topics and key points discussed
2. Important visual elements and scenes  
3. Overall narrative or message
4. Key timestamps for important moments
5. The emotional tone and atmosphere

Make the summary engaging and informative for someone who hasn't watched the video. Include specific timestamps when mentioning important moments."""

            # Call Gemini API for full summary
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            full_summary = response.text
            
            return {
                "full_summary": full_summary,
                "key_transcript_moments": key_transcript_moments,
                "key_frame_moments": key_frame_moments,
                "method": "embedding_based",
                "total_audio_segments": len(key_transcript_moments),
                "total_visual_segments": len(key_frame_moments)
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive video analysis: {str(e)}")
            # Fallback to basic content
            content = self.get_video_content(video_id)
            return {
                "full_summary": self.generate_basic_summary(video_id),
                "key_transcript_moments": content.get("transcript", [])[:5],
                "key_frame_moments": content.get("visual_descriptions", [])[:5],
                "method": "basic_fallback",
                "error": str(e)
            }
    def generate_basic_summary(self, video_id: str) -> str:
        """Generate basic summary when Gemini is not available"""
        content = self.get_video_content(video_id)
            
        summary_parts = []
        
        if content["transcript"]:
            summary_parts.append(f"Audio Content: {content['total_audio_segments']} segments processed")
            # Get first few transcript entries as preview
            preview = " | ".join([item["text"][:50] + "..." for item in content["transcript"][:3]])
            summary_parts.append(f"Transcript Preview: {preview}")
        
        if content["visual_descriptions"]:
            summary_parts.append(f"Visual Content: {content['total_visual_segments']} key frames analyzed")
            # Get first few visual descriptions as preview
            preview = " | ".join([item["description"][:50] + "..." for item in content["visual_descriptions"][:3]])
            summary_parts.append(f"Visual Preview: {preview}")
        
        if not summary_parts:
            return "No content available for summary generation."
        
        return "\n".join(summary_parts)
        
    def answer_question(self, video_id: str, question: str) -> str:
        """Answer questions about the video using Gemini API"""
        try:
            # Get video content
            content = self.get_video_content(video_id)
            
            if not content["transcript"] and not content["visual_descriptions"]:
                return "No content available to answer questions about this video."
            
            # Prepare content for Gemini
            transcript_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['text']}" for item in content["transcript"]])
            visual_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['description']}" for item in content["visual_descriptions"]])
            
            # Create question-answering prompt
            prompt = f"""Based on the following video content, please answer this question: "{question}"

AUDIO TRANSCRIPT:
{transcript_text}

VISUAL DESCRIPTIONS:
{visual_text}

Please provide a clear, accurate answer based only on the information available in the content above. If the question cannot be answered from the available content, please say so."""

            # Call Gemini API
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Sorry, I couldn't answer your question due to an error: {str(e)}"

    
# ---------------- FastAPI App ----------------
app = FastAPI(title="Enhanced Video Analysis API", version="2.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the video analytics system
video_system = VideoAnalyticsSystem()

# ---------------- API Endpoints ----------------
@app.post("/process_video")
async def process_video_endpoint(request: VideoRequest):
    """Process a video through the complete pipeline"""
    try:
        # Generate unique video ID
        import uuid
        video_id = str(uuid.uuid4())
        
        # Download video (reuse existing download logic)
        platform = identify_platform(request.video_url)
        video_path = os.path.join(WORK_DIR, f"{video_id}.mp4")
        
        # Download video asynchronously
        success = await download_video_async(request.video_url, platform, video_path)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        # Process video through the pipeline
        results = video_system.process_video(video_path, video_id)
        
        return {
            "status": "success",
            "video_id": video_id,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_video")
async def search_video_endpoint(request: dict):
    """Search processed video content"""
    try:
        query = request.get("query")
        query_type = request.get("query_type", "both")
        top_k = request.get("top_k", 5)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        results = video_system.search_video(query, query_type, top_k)
        
        return {
            "status": "success",
            "query": query,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_video")
async def summarize_video_endpoint(request: VideoRequest):
    """Summarize a video with AI-generated insights"""
    try:
        # Generate consistent video ID from URL instead of random UUID
        video_id = hashlib.md5(request.video_url.encode()).hexdigest()
        logger.info(f"=== SUMMARIZE VIDEO START ===")
        logger.info(f"Video URL: {request.video_url}")
        logger.info(f"Generated video ID: {video_id}")
        
        # Check cache FIRST - this works across sessions
        if video_id in summary_cache:
            logger.info(f" CACHE HIT: Video {video_id} found in cache")
            cached_result = summary_cache[video_id]
            logger.info(f" Cached summary length: {len(cached_result['summary'])} characters")
            logger.info(f" Cached vision API: {cached_result['vision_api_used']}")
            logger.info(f" Cached frames analyzed: {cached_result['frames_analyzed']}")
            
            return {
                "status": "success",
                "summary": cached_result["summary"],
                "platform": cached_result["platform"],
                "frames_analyzed": cached_result["frames_analyzed"],
                "video_duration": cached_result["video_duration"],
                "has_audio_transcript": cached_result["has_audio_transcript"],
                "vision_api_used": cached_result["vision_api_used"],
                "frame_coverage_info": cached_result["frame_coverage_info"],
                "cached": True
            }
        
        logger.info(f" CACHE MISS: Video {video_id} not found in cache")
        
        # Check if video file already exists locally (avoid re-download)
        platform = identify_platform(request.video_url)
        video_path = os.path.join(WORK_DIR, f"{video_id}.mp4")
        
        logger.info(f" Platform identified: {platform}")
        logger.info(f" Checking local file: {video_path}")
        
        # If video file exists and current system has processed it, use existing data
        if os.path.exists(video_path) and video_system.is_video_processed(video_id):
            logger.info(f" LOCAL HIT: Video {video_id} exists and has been processed")
            
            # Get video duration using ffprobe
            try:
                probe = ffmpeg.probe(video_path)
                video_duration = float(probe['streams'][0]['duration'])
                logger.info(f" Video duration: {video_duration} seconds")
            except Exception as e:
                logger.warning(f" Could not get video duration: {str(e)}")
                video_duration = 0
            
            # Generate intelligent summary using actual video content
            logger.info(" Attempting to generate intelligent summary...")
            try:
                logger.info("Calling video_system.generate_intelligent_summary...")
                intelligent_summary = video_system.generate_intelligent_summary(video_id)
                
                if intelligent_summary:
                    logger.info(f" Intelligent summary generated successfully!")
                    logger.info(f" Summary length: {len(intelligent_summary)} characters")
                    logger.debug(f" Summary preview: {intelligent_summary[:200]}...")
                    vision_api_used = "Gemini Pro"
                else:
                    logger.warning("  Intelligent summary returned None, falling back to basic")
                    intelligent_summary = video_system.generate_basic_summary(video_id)
                    vision_api_used = "Basic Analysis"
                    
            except Exception as e:
                logger.error(f" Intelligent summary failed: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                intelligent_summary = video_system.generate_basic_summary(video_id)
                vision_api_used = "Basic Analysis"
            
            # Get content details for metadata
            content_info = video_system.get_video_content(video_id)
            logger.info(f" Content info retrieved:")
            logger.info(f"   - Audio segments: {content_info['total_audio_segments']}")
            logger.info(f"   - Visual segments: {content_info['total_visual_segments']}")
            logger.info(f"   - Has transcript: {len(content_info['transcript']) > 0}")
            logger.info(f"   - Has visual descriptions: {len(content_info['visual_descriptions']) > 0}")
            
            frames_analyzed = 0
            if video_system.timeline and hasattr(video_system.timeline, 'frame_events'):
                frames_analyzed = len(video_system.timeline.frame_events)
            elif content_info:
                frames_analyzed = content_info.get('total_visual_segments', 0)
                
            comprehensive_analysis = video_system.get_comprehensive_video_analysis(video_id)
            # Cache and return
            result = {
                "status": "success",
                "summary": comprehensive_analysis["comprehensive_summary"],
                "platform": platform,
                "frames_analyzed": comprehensive_analysis["key_visual_moments"],
                "video_duration": video_duration,
                "has_audio_transcript": len(comprehensive_analysis["key_transcript_moments"]) > 0,
                "vision_api_used": vision_api_used,
                "frame_coverage_info": {
                    "status": "Good",
                    "coverage": "Complete"
                },
                "cached": False,
                "content_preview": {
                    "audio_segments": len(comprehensive_analysis["key_transcript_moments"]),
                    "visual_segments": len(comprehensive_analysis["key_visual_moments"]),
                    "transcript_preview": [
                        {
                            "timestamp": item["timestamp"],
                            "text": item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"]
                        } for item in comprehensive_analysis["key_transcript_moments"][:3]
                    ],
                    "visual_preview": [
                        {
                            "timestamp": item["timestamp"], 
                            "description": f"Visual moment at {item['timestamp']}s"
                        } for item in comprehensive_analysis["key_visual_moments"][:3]
                    ],
                    "method": comprehensive_analysis.get("method", "embedding_based")
                }
            }
            
            logger.info(f" Caching result for video {video_id}")
            summary_cache[video_id] = result
            save_summary_cache(summary_cache)
            logger.info("=== SUMMARIZE VIDEO COMPLETED (LOCAL) ===")
            return result
            
        
        logger.info(f" LOCAL MISS: Video {video_id} not found locally, downloading...")
        
        # Only download and process if not cached and not locally processed
        logger.info(f" Downloading video from {platform}...")
        
        # Download video
        success = await download_video_async(request.video_url, platform, video_path)
        
        if not success:
            logger.error(" Video download failed!")
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        logger.info(" Video downloaded successfully!")
        
        # Process video through the pipeline
        logger.info(" Processing video through pipeline...")
        results = video_system.process_video(video_path, video_id)
        logger.info(f" Video processing completed!")
        
        # Get video duration using ffprobe
        try:
            probe = ffmpeg.probe(video_path)
            video_duration = float(probe['streams'][0]['duration'])
            logger.info(f" Video duration: {video_duration} seconds")
        except Exception as e:
            logger.warning(f"  Could not get video duration: {str(e)}")
            video_duration = 0
        
        # Generate intelligent summary using actual video content
        logger.info(" Attempting to generate intelligent summary...")
        try:
            logger.info("Calling video_system.generate_intelligent_summary...")
            intelligent_summary = video_system.generate_intelligent_summary(video_id)
            
            if intelligent_summary:
                logger.info(f" Intelligent summary generated successfully!")
                logger.info(f" Summary length: {len(intelligent_summary)} characters")
                logger.debug(f" Summary preview: {intelligent_summary[:200]}...")
                vision_api_used = "Gemini Pro"
            else:
                logger.warning("  Intelligent summary returned None, falling back to basic")
                intelligent_summary = video_system.generate_basic_summary(video_id)
                vision_api_used = "Basic Analysis"
                
        except Exception as e:
            logger.error(f" Intelligent summary failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            intelligent_summary = video_system.generate_basic_summary(video_id)
            vision_api_used = "Basic Analysis"
        
        # Get content details for metadata
        content_info = video_system.get_video_content(video_id)
        logger.info(f" Content info retrieved:")
        logger.info(f"   - Audio segments: {content_info['total_audio_segments']}")
        logger.info(f"   - Visual segments: {content_info['total_visual_segments']}")
        logger.info(f"   - Has transcript: {len(content_info['transcript']) > 0}")
        logger.info(f"   - Has visual descriptions: {len(content_info['visual_descriptions']) > 0}")
        
        frames_analyzed = 0
        if video_system.timeline and hasattr(video_system.timeline, 'frame_events'):
            frames_analyzed = len(video_system.timeline.frame_events)
        elif content_info:
            frames_analyzed = content_info.get('total_visual_segments', 0)
        
        # Prepare result with actual content analysis
        result = {
            "status": "success",
            "summary": intelligent_summary,
            "platform": platform,
            "frames_analyzed": frames_analyzed,
            "video_duration": video_duration,
            "has_audio_transcript": len(video_system.timeline.audio_events) > 0 if video_system.timeline and hasattr(video_system.timeline, 'audio_events') else content_info.get('total_audio_segments', 0) > 0,
            "vision_api_used": vision_api_used,
            "frame_coverage_info": {
                "status": "Good",
                "coverage": "Complete"
            },
            "cached": False,
            "content_preview": {
                "audio_segments": content_info["total_audio_segments"],
                "visual_segments": content_info["total_visual_segments"],
                "transcript_preview": [item["text"][:100] + "..." for item in content_info["transcript"][:3]] if content_info["transcript"] else [],
                "visual_preview": [item["description"][:100] + "..." for item in content_info["visual_descriptions"][:3]] if content_info["visual_descriptions"] else []
            }
        }
        
        logger.info(f" Caching result for video {video_id}")
        summary_cache[video_id] = result
        save_summary_cache(summary_cache)
        logger.info("=== SUMMARIZE VIDEO COMPLETED (NEW) ===")
        return result
        
    except Exception as e:
        logger.error(f" CRITICAL ERROR in summarize_video_endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "enhanced_video_analysis",
        "version": "2.0.0"
    }

# ---------------- Helper Functions (from original) ----------------
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

async def download_video_async(url: str, platform: str, output_path: str) -> bool:
    """Download video to local file asynchronously with improved error handling and fallback options"""
    import json
    import shutil
    import glob
    
    logger.info(f"Starting async video download for {platform} from URL: {url}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Clean up any existing file
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Method 1: Try direct yt-dlp download (most reliable)
    try:
        logger.info("Method 1: Direct yt-dlp download")
        
        # Use temporary output without extension (yt-dlp will add it)
        temp_output = output_path.rsplit('.', 1)[0] + "_temp"
        
        cmd = [
            "yt-dlp",
            url,
            "-o", f"{temp_output}.%(ext)s",
            "-f", "18",  # 360p MP4 for YouTube - most reliable
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--progress",
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        ]
        
        # Platform-specific options
        if platform == 'youtube':
            cmd.extend(["--extractor-args", "youtube:player_client=android"])
        elif platform == 'tiktok':
            cmd.extend(["--user-agent", "TikTok 26.2.0 rv:262018 (iPhone; iOS 14.4.2; en_US) Cronet"])
        elif platform == 'twitter' or platform == 'x':
            cmd.extend(["--extractor-args", "twitter:api=legacy"])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
        
        if process.returncode == 0:
            # Find the downloaded file (yt-dlp adds extension)
            for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                downloaded_file = f"{temp_output}{ext}"
                if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1024:
                    shutil.move(downloaded_file, output_path)
                    logger.info(f"Download successful: {output_path} ({os.path.getsize(output_path)} bytes)")
                    return True
            
            logger.warning("yt-dlp succeeded but file not found")
        else:
            logger.warning(f"yt-dlp failed: {stderr.decode()[:200]}")
    
    except asyncio.TimeoutError:
        logger.warning("Method 1 timeout")
    except Exception as e:
        logger.warning(f"Method 1 exception: {str(e)}")
    
    # Method 2: Try with FFmpeg (fallback)
    try:
        logger.info("Method 2: FFmpeg download")
        
        # Get direct URL first
        get_url_cmd = ["yt-dlp", url, "-f", "best", "--get-url", "--no-warnings", "--quiet"]
        
        process = await asyncio.create_subprocess_exec(
            *get_url_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        
        if process.returncode == 0:
            direct_url = stdout.decode().strip().split('\n')[0]
            
            if direct_url and direct_url.startswith('http'):
                # Download with FFmpeg
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", direct_url,
                    "-c", "copy",
                    "-y",
                    "-loglevel", "warning",
                    output_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
                
                if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    logger.info(f"✓ FFmpeg download successful: {output_path}")
                    return True
                else:
                    logger.warning(f"FFmpeg failed: {stderr.decode()[:200]}")
    
    except asyncio.TimeoutError:
        logger.warning("Method 2 timeout")
    except Exception as e:
        logger.warning(f"Method 2 exception: {str(e)}")
    
    # Method 3: Try with different format (last resort)
    try:
        logger.info("Method 3: Best quality fallback")
        
        temp_output = output_path.rsplit('.', 1)[0] + "_temp3"
        
        cmd = [
            "yt-dlp",
            url,
            "-o", f"{temp_output}.%(ext)s",
            "-f", "best[ext=mp4]/best",
            "--no-playlist",
            "--quiet",
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
        
        if process.returncode == 0:
            for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
                downloaded_file = f"{temp_output}{ext}"
                if os.path.exists(downloaded_file) and os.path.getsize(downloaded_file) > 1024:
                    shutil.move(downloaded_file, output_path)
                    logger.info(f"✓ Download successful (method 3): {output_path}")
                    return True
    
    except asyncio.TimeoutError:
        logger.warning("Method 3 timeout")
    except Exception as e:
        logger.warning(f"Method 3 exception: {str(e)}")
    
    # All methods failed
    logger.error("All download methods failed")
    return False

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    logger.info("Starting Enhanced Video Analysis Web Application")
    uvicorn.run(app, host="0.0.0.0", port=9000)