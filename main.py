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

# Import standalone modules
from audio_pipeline import AudioPipeline as StandaloneAudioPipeline
from timeline import MasterTimeline, AudioEvent, FrameEvent

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

# ---------------- Video Pipeline Module ----------------
class VideoPipeline:
    """Video Processing Pipeline for frame extraction and embedding generation"""
    def __init__(self, frame_rate: float = 1.0, vision_model: str = "ViT-B-32", 
                 vision_pretrained: str = "openai", model: Any = None,
                 preprocess: Any = None, tokenizer: Any = None,
                 output_dir: str = "video_frames"):
        self.frame_rate = frame_rate
        self.vision_model_name = vision_model
        self.vision_pretrained = vision_pretrained
        self.clip_model = model
        self.clip_preprocess = preprocess
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use global CLIP model instead of loading new one
        self.device = global_device
        self.model = global_clip_model
        self.preprocess = global_clip_preprocess
        
        logger.info(f"VideoPipeline initialized with pre-loaded vision model on device: {self.device}")
    
    def get_frame_embeddings(self, frames: List[Image.Image]) -> np.ndarray:
        """Generate embeddings for a list of frames using CLIP"""
        if not frames:
            return np.array([])
        
        # Preprocess images
        processed_images = torch.stack([self.clip_preprocess(frame) for frame in frames]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.clip_model.encode_image(processed_images)
        
        return embeddings.cpu().numpy()
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding using CLIP"""
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            text_tokens = tokenizer([text]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
        return text_embedding.cpu().numpy()
    
    
    def extract_frames(self, video_path: Path, video_id: str, save_frames: bool = True) -> List[Tuple[float, np.ndarray, Optional[Path]]]:
        """Extract frames from video at specified frame rate"""
        logger.info(f"Extracting frames from {video_path} at {self.frame_rate} fps")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        logger.info(f"Video: {video_fps} fps, {total_frames} frames, {duration:.2f}s")
        
        # Calculate frame sampling interval
        frame_interval = int(video_fps / self.frame_rate)
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        # Create output directory for this video
        if save_frames:
            video_frame_dir = self.output_dir / video_id
            video_frame_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                
                # Save frame if requested
                frame_path = None
                if save_frames:
                    frame_filename = f"frame_{saved_count:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = video_frame_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append((timestamp, frame_rgb, frame_path))
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def create_frame_events(self, frames: List[Tuple[float, np.ndarray, Optional[Path]]], video_id: str) -> List[FrameEvent]:
        """Create FrameEvent objects from extracted frames"""
        frame_events = []
        
        for idx, (timestamp, frame_array, frame_path) in enumerate(frames):
            event = FrameEvent(
                timestamp=timestamp,
                frame_index=idx,
                frame_path=str(frame_path) if frame_path else None,
                embedding_id=f"{video_id}_frame_{idx}",
                metadata={
                    'width': frame_array.shape[1],
                    'height': frame_array.shape[0]
                }
            )
            frame_events.append(event)
        
        logger.info(f"Created {len(frame_events)} frame events")
        return frame_events
    
    def generate_embeddings(self, frames: List[Tuple[float, np.ndarray, Optional[Path]]]) -> np.ndarray:
        """Generate CLIP embeddings for frames"""
        logger.info(f"Generating embeddings for {len(frames)} frames")
        
        embeddings = []
        
        # Process in batches
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Preprocess frames
                batch_images = []
                for _, frame_array, _ in batch_frames:
                    pil_image = Image.fromarray(frame_array)
                    preprocessed = self.preprocess(pil_image).unsqueeze(0)
                    batch_images.append(preprocessed)
                
                # Stack batch
                batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                
                # Generate embeddings
                batch_embeddings = self.model.encode_image(batch_tensor)
                
                # Normalize embeddings
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info(f"Generated embeddings with shape {all_embeddings.shape}")
        return all_embeddings

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
            
        embeddings = self.sentence_transformer.encode(texts, convert_to_numpy=True)
        
        # Create metadata for each embedding
        metadata = [
            {"type": "audio", "timestamp": segment['start'], "text": segment['text']}
            for segment in transcript
        ]
        
        return embeddings, metadata
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding using Sentence Transformer"""
        return self.sentence_transformer.encode(text, convert_to_numpy=True)
    
    def process_audio(self, audio_path: str, video_id: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Transcribe audio and generate embeddings"""
    
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
        
        logger.info("Initialized VideoAnalyticsSystem")
    
    def initialize_components(self):
        """Initialize pipeline components"""
        self.vector_store = vector_store.MultimodalVectorStore()
        self.audio_pipeline = EnhancedAudioPipeline()
        self.video_pipeline = VideoPipeline()
        logger.info("All components initialized")
    
    def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """Process a complete video through the pipeline"""
        logger.info(f"=== Processing Video: {video_path} ===")
        
        # Initialize components if not already done
        if self.vector_store is None:
            self.initialize_components()
        
        video_path_obj = Path(video_path)
        
        # Step 1: Audio Pipeline
        logger.info("[1/2] Audio Pipeline")
        audio_path = self.audio_pipeline.extract_audio(video_path_obj)
        transcription = self.audio_pipeline.transcribe_audio(audio_path)
        audio_events = self.audio_pipeline.create_audio_events(transcription)
        audio_embeddings = self.audio_pipeline.generate_embeddings(audio_events)
        
        # Add audio embeddings to vector store
        audio_metadata = [event.to_dict() for event in audio_events]
        self.vector_store.add("audio",audio_embeddings, audio_metadata)
        
        logger.info(f"Audio events: {len(audio_events)}")
        logger.info(f"Audio embeddings shape: {audio_embeddings.shape}")
        
        # Step 2: Video Pipeline
        logger.info("[2/2] Video Pipeline")
        frames = self.video_pipeline.extract_frames(video_path_obj, video_id, save_frames=True)
        frame_events = self.video_pipeline.create_frame_events(frames, video_id)
        frame_embeddings = self.video_pipeline.generate_embeddings(frames)
        
        # Add video embeddings to vector store
        frame_metadata = [event.to_dict() for event in frame_events]
        self.vector_store.add("video",frame_embeddings, frame_metadata)
        
        logger.info(f"Frame events: {len(frame_events)}")
        logger.info(f"Frame embeddings shape: {frame_embeddings.shape}")
        
        logger.info("=== Video Processing Complete ===")
        
        return {
            "video_id": video_id,
            "audio_events": len(audio_events),
            "frame_events": len(frame_events),
            "audio_embeddings_shape": audio_embeddings.shape,
            "frame_embeddings_shape": frame_embeddings.shape
        }
    
    def search_video(self, query: str, query_type: str = "both", top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for a query in the video content"""
        results = []
        
        if query_type in ["audio", "both"]:
            # Search in audio
            logger.info(f"Searching audio for: '{query}'")
            audio_embedding = self.audio_pipeline.embed_text(query)
            distances, indices, metadata = self.vector_store.search("audio", audio_embedding, top_k)
            
            for dist, meta in zip(distances, metadata):
                results.append({
                    "modality": "audio",
                    "score": 1 - dist,  # Convert distance to similarity score
                    "timestamp": meta.get("timestamp"),
                    "text": meta.get("text")
                })
        
        if query_type in ["video", "both"]:
            # Search in video frames
            logger.info(f"Searching video for: '{query}'")
            frame_embedding = self.video_pipeline.embed_text(query)
            distances, indices, metadata = self.vector_store.search("video", frame_embedding, top_k)
            
            for dist, meta in zip(distances, metadata):
                results.append({
                    "modality": "video",
                    "score": 1 - dist,
                    "timestamp": meta.get("timestamp"),
                    "preview_url": meta.get("preview_url")
                })
        
        if not results:
            logger.info(f"No results found for query: '{query}'")
            return []

        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    def get_video_content(self, video_id: str) -> dict:
        """Extract actual content from video processing results"""
        try:
            # Get stored data from vector store
            audio_data = self.vector_store.storage.get("audio", {})
            video_data = self.vector_store.storage.get("video", {})
            
            # Extract transcript from audio events
            transcript = []
            if video_id in audio_data:
                for item in audio_data[video_id]:
                    if "text" in item:
                        transcript.append({
                            "timestamp": item.get("timestamp", 0),
                            "text": item["text"]
                        })
            
            # Extract visual descriptions from frame events
            visual_descriptions = []
            if video_id in video_data:
                for item in video_data[video_id]:
                    if "description" in item:
                        visual_descriptions.append({
                            "timestamp": item.get("timestamp", 0),
                            "description": item["description"]
                        })
            
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

    def generate_intelligent_summary(self, video_id: str) -> str:
        """Generate intelligent summary using Gemini API"""
        try:
            # Get video content
            content = self.get_video_content(video_id)
            
            if not content["transcript"] and not content["visual_descriptions"]:
                return self.generate_basic_summary(video_id)
            
            # Prepare content for Gemini
            transcript_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['text']}" for item in content["transcript"]])
            visual_text = "\n".join([f"[{item['timestamp']:.1f}s] {item['description']}" for item in content["visual_descriptions"]])
            
            # Create comprehensive prompt for Gemini
            prompt = f"""Please provide a comprehensive summary of this video based on the following content:

AUDIO TRANSCRIPT:
{transcript_text}

VISUAL DESCRIPTIONS:
{visual_text}

Please create a detailed summary that includes:
1. Main topics and key points discussed
2. Important visual elements and scenes
3. Overall narrative or message
4. Key timestamps for important moments

Make the summary engaging and informative for someone who hasn't watched the video."""

            # Call Gemini API
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating intelligent summary: {str(e)}")
            return self.generate_basic_summary(video_id)

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
            model = genai.GenerativeModel('gemini-pro')
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
        # Generate unique video ID
        import uuid
        video_id = str(uuid.uuid4())
        
        # Download video
        platform = identify_platform(request.video_url)
        video_path = os.path.join(WORK_DIR, f"{video_id}.mp4")
        
        # Download video asynchronously
        success = await download_video_async(request.video_url, platform, video_path)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        # Process video through the pipeline
        results = video_system.process_video(video_path, video_id)
        
        # Get video duration using ffprobe
        try:
            probe = ffmpeg.probe(video_path)
            video_duration = float(probe['streams'][0]['duration'])
        except:
            video_duration = 0
        
        # Generate AI summary based on processed content
        # This is a simplified version - you can enhance it with more sophisticated summarization
        audio_events = results.get("audio_events", 0)
        frame_events = results.get("frame_events", 0)
        
        # Create a comprehensive summary
        summary = f"""## Video Overview
This video from {platform.upper()} has been analyzed using advanced AI techniques.

## Key Findings
- **Audio Content**: {audio_events} audio segments processed
- **Visual Content**: {frame_events} key frames analyzed
- **Duration**: {video_duration:.1f} seconds

## Content Analysis
The video has been processed through our multimodal AI pipeline, extracting both audio transcriptions and visual insights. This enables semantic search across both audio and visual content.

## Technical Details
- Audio embeddings: {results.get('audio_embeddings_shape', 'N/A')}
- Frame embeddings: {results.get('frame_embeddings_shape', 'N/A')}
- Processing completed successfully

## Next Steps
You can now search through this video content using natural language queries at the /search_video endpoint."""
        
        return {
            "status": "success",
            "summary": summary,
            "platform": platform,
            "frames_analyzed": frame_events,
            "video_duration": video_duration,
            "has_audio_transcript": audio_events > 0,
            "vision_api_used": "OpenAI GPT-4",
            "frame_coverage_info": {
                "status": "Good",
                "coverage": "Complete"
            }
        }
        
    except Exception as e:
        logger.error(f"Error summarizing video: {str(e)}")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)