"""
Query and Analysis Module
Handles user queries, vector search, and LLM-based reasoning
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from loguru import logger

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from settings import settings
from timeline import MasterTimeline, TimeRange
from vector_store import MultimodalVectorStore
from sentence_transformers import SentenceTransformer
import open_clip


class QueryAnalyzer:
    """
    Query and Analysis System
    
    Pipeline:
    1. Query embedding
    2. Vector search (Audio + Vision)
    3. Retrieve top-K relevant time ranges & frames
    4. Send to LLM (Gemini/Claude) for reasoning
    """
    
    def __init__(
        self,
        vector_store: MultimodalVectorStore,
        timeline: MasterTimeline,
        llm_provider: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.timeline = timeline
        self.llm_provider = llm_provider or settings.LLM_PROVIDER
        
        # Load embedding models (same as used for indexing)
        logger.info("Loading query embedding models")
        self.text_embedder = SentenceTransformer(settings.AUDIO_EMBED_MODEL)
        
        # Load vision model for image queries
        self.vision_model, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            settings.VISION_MODEL,
            pretrained=settings.VISION_PRETRAINED,
            device=settings.DEVICE
        )
        self.vision_model.eval()
        
        # Initialize LLM client
        self._init_llm_client()
    
    def _init_llm_client(self) -> None:
        """Initialize LLM client based on provider"""
        if self.llm_provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai not installed")
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.llm_client = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Initialized Gemini LLM client")
        
        elif self.llm_provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic not installed")
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.llm_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("Initialized Anthropic LLM client")
        
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def embed_text_query(self, query: str) -> np.ndarray:
        """Embed text query for search"""
        return self.text_embedder.encode([query], convert_to_numpy=True)[0]
    
    def embed_image_query(self, image_path: Path) -> np.ndarray:
        """Embed image query for search"""
        import torch
        
        image = Image.open(image_path)
        preprocessed = self.vision_preprocess(image).unsqueeze(0).to(settings.DEVICE)
        
        with torch.no_grad():
            embedding = self.vision_model.encode_image(preprocessed)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0]
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        modality: str = "both"
    ) -> Dict[str, Any]:
        """
        Search for relevant content using query
        
        Args:
            query: Text query
            top_k: Number of results to return
            modality: "audio", "video", or "both"
        
        Returns:
            Dict with search results
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Searching for query: '{query}' (modality: {modality})")
        
        # Embed query
        text_embedding = self.embed_text_query(query)
        
        if modality == "both":
            # Search both modalities
            results = self.vector_store.search_both(
                audio_query=text_embedding,
                video_query=text_embedding,
                top_k=top_k
            )
        elif modality == "audio":
            distances, indices, metadata = self.vector_store.search_multimodal(
                text_embedding, "audio", top_k
            )
            results = {
                "audio": {
                    "distances": distances,
                    "indices": indices,
                    "metadata": metadata
                }
            }
        elif modality == "video":
            distances, indices, metadata = self.vector_store.search_multimodal(
                text_embedding, "video", top_k
            )
            results = {
                "video": {
                    "distances": distances,
                    "indices": indices,
                    "metadata": metadata
                }
            }
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        return results
    
    def get_time_ranges_from_results(
        self,
        results: Dict[str, Any],
        window_seconds: float = 5.0
    ) -> List[TimeRange]:
        """
        Extract time ranges from search results
        
        Args:
            results: Search results
            window_seconds: Time window around each result
        
        Returns:
            List of TimeRange objects
        """
        time_ranges = []
        
        # Collect timestamps from all modalities
        timestamps = []
        
        if "audio" in results:
            for meta in results["audio"]["metadata"]:
                timestamps.append(meta["timestamp"])
        
        if "video" in results:
            for meta in results["video"]["metadata"]:
                timestamps.append(meta["timestamp"])
        
        # Create time ranges with windows
        for ts in timestamps:
            time_range = TimeRange(
                start_seconds=max(0, ts - window_seconds),
                end_seconds=min(self.timeline.duration_seconds, ts + window_seconds)
            )
            time_ranges.append(time_range)
        
        # Merge overlapping ranges
        time_ranges = self._merge_time_ranges(time_ranges)
        
        return time_ranges
    
    def _merge_time_ranges(self, ranges: List[TimeRange]) -> List[TimeRange]:
        """Merge overlapping time ranges"""
        if not ranges:
            return []
        
        # Sort by start time
        sorted_ranges = sorted(ranges, key=lambda r: r.start_seconds)
        
        merged = [sorted_ranges[0]]
        
        for current in sorted_ranges[1:]:
            last = merged[-1]
            
            if current.start_seconds <= last.end_seconds:
                # Overlapping, merge
                merged[-1] = TimeRange(
                    start_seconds=last.start_seconds,
                    end_seconds=max(last.end_seconds, current.end_seconds)
                )
            else:
                # Non-overlapping, add new range
                merged.append(current)
        
        return merged
    
    def prepare_llm_context(
        self,
        time_ranges: List[TimeRange],
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Prepare context for LLM from time ranges
        
        Args:
            time_ranges: List of relevant time ranges
            max_frames: Maximum number of frames to include
        
        Returns:
            Dict with transcript snippets, timestamps, and frame paths
        """
        max_frames = max_frames or settings.MAX_FRAMES_TO_LLM
        
        context = {
            "transcript_snippets": [],
            "timestamps": [],
            "frames": []
        }
        
        for time_range in time_ranges:
            # Get transcript for this range
            transcript = self.timeline.get_audio_transcript(time_range)
            context["transcript_snippets"].append({
                "time_range": time_range.to_dict(),
                "text": transcript
            })
            
            # Get frames for this range
            frames = self.timeline.get_frames_in_range(time_range, max_frames=max_frames)
            for frame in frames:
                if frame.frame_path:
                    context["frames"].append({
                        "timestamp": frame.timestamp,
                        "path": frame.frame_path,
                        "frame_index": frame.frame_index
                    })
        
        # Limit total frames
        if len(context["frames"]) > max_frames:
            # Sample evenly
            step = len(context["frames"]) / max_frames
            context["frames"] = [
                context["frames"][int(i * step)]
                for i in range(max_frames)
            ]
        
        return context
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for LLM"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def query_llm(
        self,
        query: str,
        context: Dict[str, Any],
        task: str = "answer_question"
    ) -> str:
        """
        Query LLM with context
        
        Args:
            query: User query
            context: Prepared context (transcripts, frames)
            task: Type of task ("answer_question", "detect_events", "summarize")
        
        Returns:
            LLM response
        """
        logger.info(f"Querying LLM with task: {task}")
        
        # Prepare prompt
        prompt = self._build_prompt(query, context, task)
        
        # Query based on provider
        if self.llm_provider == "gemini":
            return self._query_gemini(prompt, context)
        elif self.llm_provider == "anthropic":
            return self._query_anthropic(prompt, context)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def _build_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        task: str
    ) -> str:
        """Build prompt for LLM"""
        prompt_parts = []
        
        # Task instruction
        if task == "answer_question":
            prompt_parts.append(
                "You are analyzing a video. Answer the user's question based on the provided context."
            )
        elif task == "detect_events":
            prompt_parts.append(
                "You are analyzing a video. Detect and describe important events based on the provided context."
            )
        elif task == "summarize":
            prompt_parts.append(
                "You are analyzing a video. Provide a comprehensive summary based on the provided context."
            )
        
        # Add transcript context
        if context.get("transcript_snippets"):
            prompt_parts.append("\n## Transcript Snippets:")
            for snippet in context["transcript_snippets"]:
                time_range = snippet["time_range"]
                prompt_parts.append(
                    f"\n[{time_range['start_seconds']:.2f}s - {time_range['end_seconds']:.2f}s]: {snippet['text']}"
                )
        
        # Add frame information
        if context.get("frames"):
            prompt_parts.append(f"\n## Visual Frames: {len(context['frames'])} frames provided")
            for i, frame in enumerate(context["frames"]):
                prompt_parts.append(f"Frame {i+1}: timestamp {frame['timestamp']:.2f}s")
        
        # Add user query
        prompt_parts.append(f"\n## User Query:\n{query}")
        
        prompt_parts.append("\n## Your Response:")
        
        return "\n".join(prompt_parts)
    
    def _query_gemini(self, prompt: str, context: Dict[str, Any]) -> str:
        """Query Gemini LLM"""
        # Prepare content with images
        content = [prompt]
        
        # Add images if available
        for frame in context.get("frames", [])[:settings.MAX_FRAMES_TO_LLM]:
            if frame.get("path"):
                try:
                    img = Image.open(frame["path"])
                    content.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {frame['path']}: {e}")
        
        # Generate response
        response = self.llm_client.generate_content(
            content,
            generation_config={
                "max_output_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE
            }
        )
        
        return response.text
    
    def _query_anthropic(self, prompt: str, context: Dict[str, Any]) -> str:
        """Query Anthropic Claude"""
        # Prepare messages with images
        messages_content = []
        
        # Add images first
        for frame in context.get("frames", [])[:settings.MAX_FRAMES_TO_LLM]:
            if frame.get("path"):
                try:
                    img_b64 = self._image_to_base64(frame["path"])
                    messages_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to load image {frame['path']}: {e}")
        
        # Add text prompt
        messages_content.append({
            "type": "text",
            "text": prompt
        })
        
        # Query Claude
        response = self.llm_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            messages=[{
                "role": "user",
                "content": messages_content
            }]
        )
        
        return response.content[0].text
    
    def analyze(
        self,
        query: str,
        top_k: Optional[int] = None,
        window_seconds: float = 5.0,
        task: str = "answer_question"
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline
        
        Args:
            query: User query
            top_k: Number of search results
            window_seconds: Time window for context
            task: Analysis task type
        
        Returns:
            Dict with search results, context, and LLM response
        """
        logger.info(f"Running complete analysis for query: '{query}'")
        
        # Step 1: Vector search
        search_results = self.search(query, top_k=top_k)
        
        # Step 2: Extract time ranges
        time_ranges = self.get_time_ranges_from_results(
            search_results,
            window_seconds=window_seconds
        )
        
        # Step 3: Prepare LLM context
        llm_context = self.prepare_llm_context(time_ranges)
        
        # Step 4: Query LLM
        llm_response = self.query_llm(query, llm_context, task=task)
        
        return {
            "query": query,
            "search_results": search_results,
            "time_ranges": [tr.to_dict() for tr in time_ranges],
            "context": llm_context,
            "response": llm_response
        }