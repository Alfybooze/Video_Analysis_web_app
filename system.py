"""
Video Analytics System - Main Orchestrator
Coordinates all pipeline components
"""
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import sys

from settings import settings
from ingestion import VideoIngestion
from timeline import MasterTimeline
from audio_pipeline import AudioPipeline
from video_pipeline import VideoPipeline
from vector_store import MultimodalVectorStore
from query_analyzer import QueryAnalyzer


class VideoAnalyticsSystem:
    """
    Complete Video Analytics System
    
    Orchestrates the entire pipeline:
    1. Video Ingestion → Master Timeline
    2. Audio Pipeline → Transcript + Embeddings
    3. Video Pipeline → Frames + Embeddings
    4. Vector Store → Index embeddings
    5. Query Analyzer → Search + LLM reasoning
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the video analytics system
        
        Args:
            data_dir: Base directory for data storage
            llm_provider: LLM provider ("gemini" or "anthropic")
        """
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_provider = llm_provider or settings.LLM_PROVIDER
        
        # Components (initialized on-demand)
        self.ingestion = None
        self.audio_pipeline = None
        self.video_pipeline = None
        self.vector_store = None
        self.query_analyzer = None
        
        # Current video state
        self.video_id = None
        self.video_path = None
        self.timeline = None
        
        logger.info(f"Initialized VideoAnalyticsSystem with data_dir: {self.data_dir}")
    
    def _init_components(self):
        """Initialize pipeline components lazily"""
        if self.ingestion is None:
            self.ingestion = VideoIngestion()
        
        if self.audio_pipeline is None:
            self.audio_pipeline = AudioPipeline()
        
        if self.video_pipeline is None:
            self.video_pipeline = VideoPipeline()
        
        if self.vector_store is None:
            self.vector_store = MultimodalVectorStore()
    
    def process_video(
        self,
        video_url: str,
        save_frames: bool = True,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process a complete video through the pipeline
        
        Args:
            video_url: URL or path to video
            save_frames: Whether to save extracted frames
            force_reprocess: Force reprocessing even if cached
        
        Returns:
            Dict with processing results
        """
        logger.info(f"=== Processing Video: {video_url} ===")
        
        self._init_components()
        
        # Step 1: Video Ingestion
        logger.info("[1/4] Video Ingestion")
        self.video_id, self.video_path, self.timeline = self.ingestion.ingest(
            video_url,
            force_download=force_reprocess
        )
        
        logger.info(f"Video ID: {self.video_id}")
        logger.info(f"Duration: {self.timeline.duration_seconds:.2f}s")
        
        # Step 2: Audio Pipeline
        logger.info("[2/4] Audio Pipeline")
        audio_events, audio_embeddings = self.audio_pipeline.process(
            self.video_path,
            self.timeline
        )
        
        logger.info(f"Audio events: {len(audio_events)}")
        logger.info(f"Audio embeddings shape: {audio_embeddings.shape}")
        
        # Step 3: Video Pipeline
        logger.info("[3/4] Video Pipeline")
        frame_events, frame_embeddings = self.video_pipeline.process(
            self.video_path,
            self.video_id,
            self.timeline,
            save_frames=save_frames
        )
        
        logger.info(f"Frame events: {len(frame_events)}")
        logger.info(f"Frame embeddings shape: {frame_embeddings.shape}")
        
        # Step 4: Build Vector Store
        logger.info("[4/4] Building Vector Store")
        self.vector_store.add_audio(audio_embeddings, audio_events)
        self.vector_store.add_video(frame_embeddings, frame_events)
        
        # Save everything
        self._save_state()
        
        logger.info("=== Video Processing Complete ===")
        
        return {
            "video_id": self.video_id,
            "duration": self.timeline.duration_seconds,
            "audio_events": len(audio_events),
            "frame_events": len(frame_events),
            "timeline": self.timeline.to_dict()
        }
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        window_seconds: float = 5.0,
        task: str = "answer_question"
    ) -> Dict[str, Any]:
        """
        Query the processed video
        
        Args:
            query: User query text
            top_k: Number of search results
            window_seconds: Time window for context
            task: Analysis task ("answer_question", "detect_events", "summarize")
        
        Returns:
            Dict with analysis results
        """
        if self.timeline is None:
            raise ValueError("No video processed. Call process_video() first.")
        
        # Initialize query analyzer if needed
        if self.query_analyzer is None:
            self.query_analyzer = QueryAnalyzer(
                self.vector_store,
                self.timeline,
                llm_provider=self.llm_provider
            )
        
        logger.info(f"Querying: '{query}'")
        
        # Run analysis
        results = self.query_analyzer.analyze(
            query,
            top_k=top_k,
            window_seconds=window_seconds,
            task=task
        )
        
        return results
    
    def detect_events(self, top_k: int = 20) -> Dict[str, Any]:
        """
        Detect important events in the video
        
        Args:
            top_k: Number of events to detect
        
        Returns:
            Dict with detected events
        """
        return self.query(
            "What are the main events or important moments in this video?",
            top_k=top_k,
            task="detect_events"
        )
    
    def summarize(self, top_k: int = 30) -> Dict[str, Any]:
        """
        Generate a summary of the video
        
        Args:
            top_k: Number of segments to include
        
        Returns:
            Dict with summary
        """
        return self.query(
            "Provide a comprehensive summary of this video.",
            top_k=top_k,
            task="summarize"
        )
    
    def _save_state(self):
        """Save system state to disk"""
        if self.video_id is None:
            return
        
        video_data_dir = self.data_dir / self.video_id
        video_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save timeline
        timeline_path = video_data_dir / "timeline.json"
        self.timeline.save_json(str(timeline_path))
        logger.info(f"Saved timeline to {timeline_path}")
        
        # Save vector store
        vector_store_path = video_data_dir / "vector_store"
        self.vector_store.save(vector_store_path)
        logger.info(f"Saved vector store to {vector_store_path}")
    
    def load_state(self, video_id: str):
        """
        Load previously processed video state
        
        Args:
            video_id: Video ID to load
        """
        self._init_components()
        
        video_data_dir = self.data_dir / video_id
        
        if not video_data_dir.exists():
            raise ValueError(f"No data found for video_id: {video_id}")
        
        # Load timeline
        timeline_path = video_data_dir / "timeline.json"
        self.timeline = MasterTimeline.load_json(str(timeline_path))
        self.video_id = video_id
        logger.info(f"Loaded timeline: {self.timeline}")
        
        # Load vector store
        vector_store_path = video_data_dir / "vector_store"
        self.vector_store.load(vector_store_path)
        logger.info(f"Loaded vector store from {vector_store_path}")
        
        # Reset query analyzer to use new data
        self.query_analyzer = None
    
    def get_transcript(self, with_timestamps: bool = False) -> str:
        """
        Get the complete video transcript
        
        Args:
            with_timestamps: Include timestamps
        
        Returns:
            Transcript text
        """
        if self.timeline is None:
            raise ValueError("No video processed")
        
        if with_timestamps:
            return self.audio_pipeline.get_transcript_with_timestamps(
                self.timeline.audio_events
            )
        else:
            return self.audio_pipeline.get_transcript_text(
                self.timeline.audio_events
            )
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get information about the current video"""
        if self.timeline is None:
            raise ValueError("No video processed")
        
        return {
            "video_id": self.video_id,
            "duration": self.timeline.duration_seconds,
            "metadata": self.timeline.metadata,
            "audio_events": len(self.timeline.audio_events),
            "frame_events": len(self.timeline.frame_events)
        }


def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level
    )
    logger.add(
        Path(settings.DATA_DIR) / "logs" / "video_analytics_{time}.log",
        rotation="100 MB",
        retention="30 days",
        level=level
    )


# Convenience function
def create_system(**kwargs) -> VideoAnalyticsSystem:
    """
    Create a VideoAnalyticsSystem instance
    
    Usage:
        system = create_system()
        system.process_video("path/to/video.mp4")
        result = system.query("What happens at the beginning?")
    """
    setup_logging()
    return VideoAnalyticsSystem(**kwargs)