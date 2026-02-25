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
from typing import Any, List
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
        self._init_gemini_vision_client()
    
    def _init_llm_client(self) -> None:
        """Initialize LLM client based on provider"""
        if self.llm_provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai not installed")
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.llm_client = genai.GenerativeModel('gemini-2.5-flash')
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

    def _init_gemini_vision_client(self) -> None:
        """Initialize dedicated Gemini client for vision tasks"""
        if genai is None:
            logger.warning("google-generativeai not installed, visual descriptions will be disabled")
            self.gemini_vision_client = None
            return
            
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set, visual descriptions will be disabled")
            self.gemini_vision_client = None
            return
            
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_vision_client = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Initialized dedicated Gemini vision client")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini vision client: {e}")
            self.gemini_vision_client = None
    def embed_text_query(self, query: str) -> np.ndarray:
        """Embed text query for search"""
        return self.text_embedder.encode([query], convert_to_numpy=True)[0]
    
    def embed_image_query(self, image_path: Path) -> np.ndarray:
        """Embed image query for search"""
        import torch
        import numpy as np
        
        image = Image.open(image_path)
        
        # Apply preprocessing - vision_preprocess is a tuple of transforms
        # Use the validation transform (index 1) for inference
        if isinstance(self.vision_preprocess, tuple):
            preprocess_transform = self.vision_preprocess[1]  # validation transform
        else:
            preprocess_transform = self.vision_preprocess
        
        # Apply the transform
        preprocessed = preprocess_transform(image)
        
        # Convert to tensor if it's a numpy array
        if isinstance(preprocessed, np.ndarray):
            preprocessed = torch.from_numpy(preprocessed)
        elif not isinstance(preprocessed, torch.Tensor):
            # If it's neither numpy nor tensor, convert to tensor
            preprocessed = torch.tensor(preprocessed)
        
        # Add batch dimension and move to device
        if preprocessed.dim() == 3:  # If no batch dimension
            preprocessed = preprocessed.unsqueeze(0)
        
        preprocessed = preprocessed.to(settings.DEVICE)
        
        with torch.no_grad():
            embedding = self.vision_model.encode_image(preprocessed)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        # Ensure we return a numpy array, not a tensor
        result = embedding.cpu().numpy()[0]
        return result
    
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
            # Search all modalities
            results = self.vector_store.search_all(text_embedding, top_k)
        elif modality == "audio":
            audio_results = self.vector_store.search("audio", text_embedding, top_k)
            results = {
                "audio": audio_results
            }
        elif modality == "video":
            video_results = self.vector_store.search("video", text_embedding, top_k)
            results = {
                "video": video_results
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
    def _generate_visual_description(self, frames: List[Dict[str, Any]]) -> str:
        """
        Generate a visual description for a list of frames using Gemini.
        
        Args:
            frames: List of frame objects with 'path' and 'timestamp'.
            
        Returns:
            A string describing the visual content of the frames.
        """
        if not frames:
            return "No frames available for this segment."

        # Prepare the prompt for the vision model
        prompt = "Describe the key visual elements and actions across these frames in a single, concise sentence."
        
        # Select a subset of frames to pass to the model
        max_frames_to_llm = min(len(frames), settings.MAX_FRAMES_TO_LLM)
        indices = np.linspace(0, len(frames) - 1, num=max_frames_to_llm, dtype=int)
        selected_frames = [frames[i] for i in indices]

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
        # Call the Gemini API using dedicated vision client
        try:
            if self.gemini_vision_client is None:
                logger.warning("Gemini vision client not available, skipping visual description")
                return "Visual description unavailable - Gemini client not initialized."
            
            response = self.gemini_vision_client.generate_content(model_input)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating visual description with Gemini: {e}")
            return "Error generating visual description."
        
    def get_intelligent_time_ranges_from_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        max_segments: int = 5,
        similarity_threshold: float = 0.7,
        time_window_seconds: Optional[float] = None
    ) -> List[TimeRange]:
        """
        Intelligently select relevant time ranges using CLIP embeddings based on query similarity.
        
        Args:
            results: List of search results from vector store
            query: User query string
            max_segments: Maximum number of segments to return
            similarity_threshold: Minimum similarity score to consider a segment relevant
            time_window_seconds: Optional time window for local queries (e.g., "what happens at 4 minutes?")
            
        Returns:
            List of TimeRange objects containing the most relevant segments
        """
        logger.info(f"Intelligent segment selection for query: '{query}'")
        
        if not results:
            return []
            
        # Embed the query using CLIP text encoder
        try:
            query_embedding = self._embed_text(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            # Fallback to original method
            fallback_ranges = []
            for result in results[:max_segments]:
                time_range = TimeRange(
                    start_seconds=result.get('start_seconds', 0),
                    end_seconds=result.get('end_seconds', 0)
                )
                fallback_ranges.append(time_range)
            return fallback_ranges
        
        # Analyze query type for targeted search
        query_lower = query.lower()
        is_global_question = any(keyword in query_lower for keyword in [
            "what is this video about", "summary", "overview", "what happens", 
            "main topic", "general idea", "content"
        ])
        
        is_local_question = any(keyword in query_lower for keyword in [
            "at 4 minutes", "at 5 minutes", "at time", "when does", 
            "specific moment", "particular scene"
        ])
        
        is_event_detection = any(keyword in query_lower for keyword in [
            "comparison starts", "event", "important moment", "key scene", 
            "significant", "highlight", "crucial"
        ])
        
        # Process results with semantic scoring
        scored_segments = []
        
        for result in results:
            try:
                # Extract time range and metadata
                time_range = TimeRange(
                    start_seconds=result.get('start_seconds', 0),
                    end_seconds=result.get('end_seconds', 0)
                )
                
                # Get similarity score from result
                similarity_score = result.get('similarity', 0.0)
                
                # Apply query-type specific scoring
                if is_global_question:
                    # Boost scores for segments that represent core content
                    # Look for segments with high semantic similarity and good coverage
                    content_score = self._score_global_relevance(result, time_range)
                    similarity_score *= content_score
                    
                elif is_local_question and time_window_seconds:
                    # Boost scores for segments within the specified time window
                    time_score = self._score_time_proximity(time_range, time_window_seconds)
                    similarity_score *= time_score
                    
                elif is_event_detection:
                    # Look for semantic spikes - segments with high distinctiveness
                    event_score = self._score_event_significance(result, time_range)
                    similarity_score *= event_score
                
                # Filter by threshold
                if similarity_score >= similarity_threshold:
                    scored_segments.append({
                        'time_range': time_range,
                        'score': similarity_score,
                        'result': result
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing result: {e}")
                continue
        
        # Sort by score and select top segments
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply intelligent selection strategy
        selected_segments = []
        used_time_ranges = []
        
        for segment in scored_segments[:max_segments * 2]:  # Consider more candidates
            time_range = segment['time_range']
            
            # Avoid overlapping segments (prefer higher-scoring ones)
            if not self._overlaps_with_existing(time_range, used_time_ranges):
                selected_segments.append(time_range)
                used_time_ranges.append(time_range)
                
                if len(selected_segments) >= max_segments:
                    break
        
        # Fallback if no segments meet threshold
        if not selected_segments:
            logger.warning("No segments met similarity threshold, using top results")
            fallback_ranges = []
            for result in results[:max_segments]:
                time_range = TimeRange(
                    start_seconds=result.get('start_seconds', 0),
                    end_seconds=result.get('end_seconds', 0)
                )
                fallback_ranges.append(time_range)
            return fallback_ranges
        
        logger.info(f"Selected {len(selected_segments)} intelligent segments")
        return selected_segments
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using CLIP text encoder"""
        # Use the existing text embedding functionality
        if hasattr(self, 'text_embedder') and self.text_embedder:
            return self.text_embedder.encode([text], convert_to_numpy=True)[0]
        else:
            # Fallback - return a zero vector if no text embedder is available
            logger.warning("No text embedder available, returning zero vector")
            return np.zeros(512)  # Assuming 512-dimensional embeddings
        
    def _score_global_relevance(self, result: Dict[str, Any], time_range: TimeRange) -> float:
        """Score segments for global questions (summary/overview)"""
        score = 1.0
        
        # Boost segments with longer duration (likely more substantial content)
        duration = time_range.end_seconds - time_range.start_seconds
        if duration > 10:  # Prefer segments longer than 10 seconds
            score *= 1.2
        
        # Boost segments with both audio and visual content
        if result.get('has_audio') and result.get('has_visual'):
            score *= 1.3
            
        # Boost segments that appear early in the video (often contain key info)
        if time_range.start_seconds < 60:  # First minute
            score *= 1.1
            
        return score
    
    def _score_time_proximity(self, time_range: TimeRange, target_time: float) -> float:
        """Score segments based on proximity to target time"""
        segment_center = (time_range.start_seconds + time_range.end_seconds) / 2
        distance = abs(segment_center - target_time)
        
        # Exponential decay based on distance (closer = higher score)
        if distance < 30:  # Within 30 seconds
            return 1.5
        elif distance < 60:  # Within 1 minute
            return 1.2
        elif distance < 120:  # Within 2 minutes
            return 1.0
        else:
            return 0.8
    
    def _score_event_significance(self, result: Dict[str, Any], time_range: TimeRange) -> float:
        """Score segments for event detection questions"""
        score = 1.0
        
        # Look for segments with high semantic distinctiveness
        if result.get('similarity', 0) > 0.8:
            score *= 1.4  # High similarity indicates strong relevance
            
        # Boost segments with visual changes (scene transitions)
        if result.get('visual_change_score', 0) > 0.5:
            score *= 1.2
            
        # Prefer segments that don't overlap too much with neighbors
        # (indicating distinct events)
        return score
    
    def _overlaps_with_existing(self, time_range: TimeRange, existing_ranges: List[TimeRange]) -> bool:
        """Check if time range overlaps with already selected ranges"""
        for existing in existing_ranges:
            # Check for overlap (simplified)
            if (time_range.start_seconds < existing.end_seconds and 
                time_range.end_seconds > existing.start_seconds):
                return True
        return False
    
    def _analyze_time_segment(
        self,
        time_range: TimeRange,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single time segment for content preview.
        
        Args:
            time_range: The time range to analyze.
            max_frames: Max frames for visual preview.
            
        Returns:
            A dictionary with preview content.
        """
        max_frames = max_frames or settings.MAX_FRAMES_TO_LLM
        
        # Get audio and visual data for the segment
        audio_segments = self.timeline.get_audio_transcript(time_range)
        visual_segments = self.timeline.get_frames_in_range(time_range, max_frames=max_frames)
        
        frame_dicts = []
        for frame_event in visual_segments:
            frame_dict = {
                'path': frame_event.frame_path,
                'timestamp': frame_event.timestamp
            }
            frame_dicts.append(frame_dict)
        
        # Generate visual description for the entire segment
        visual_description = self._generate_visual_description(frame_dicts)
        
        # Create transcript preview
        transcript_preview = [audio_segments] if audio_segments else []
        
        # Create visual preview (now with descriptions)
        visual_preview = [
            f"Segment {time_range.start_seconds:.1f}s-{time_range.end_seconds:.1f}s: {visual_description}"
        ]

        return {
            "audio_segments": audio_segments,
            "visual_segments": visual_segments,
            "transcript_preview": transcript_preview,
            "visual_preview": visual_preview
        }
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
        
        # Fixed: Handle different response types
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            return response.parts[0].text if hasattr(response.parts[0], 'text') else str(response.parts[0])
        else:
            return str(response)
    
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
    
    def analyze(self,query: str,top_k: Optional[int] = None, window_seconds: float = 5.0,task: str = "answer_question") -> Dict[str, Any]:
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
        all_results = []
        
        if "audio" in search_results:
            for i, (distance, idx, metadata) in enumerate(zip(
                search_results["audio"]["distances"],
                search_results["audio"]["indices"], 
                search_results["audio"]["metadata"]
            )):
                all_results.append({
                    "modality": "audio",
                    "distance": distance,
                    "index": idx,
                    "metadata": metadata,
                    "rank": i
                })
    
    # Extract video results
        if "video" in search_results:
            for i, (distance, idx, metadata) in enumerate(zip(
                search_results["video"]["distances"],
                search_results["video"]["indices"],
                search_results["video"]["metadata"]
            )):
                all_results.append({
                    "modality": "video", 
                    "distance": distance,
                    "index": idx,
                    "metadata": metadata,
                    "rank": i
                })
        
        # Step 2: Extract time ranges
        # Use intelligent segment selection instead of basic time range extraction
        time_ranges = self.get_intelligent_time_ranges_from_results(
            all_results, 
            query,
            max_segments=5,
            similarity_threshold=0.7,
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