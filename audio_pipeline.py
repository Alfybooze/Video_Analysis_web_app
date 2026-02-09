"""
Audio Pipeline
- Extracts audio from video
- Performs speech-to-text (ASR) using Whisper
- Creates timestamped transcript
- Generates audio embeddings
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import ffmpeg
import whisper
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Union
from settings import settings
from timeline import MasterTimeline, AudioEvent


class AudioPipeline:
    """
    Audio Processing Pipeline
    
    FFmpeg → Audio Stream → Speech-to-Text (ASR) → Timestamped Transcript → Audio Embeddings
    """
    
    def __init__(
        self,
        whisper_model: Optional[Union[str, whisper.model.Whisper]] = "base",
        embed_model: Optional[Union[str, SentenceTransformer]] = "all-MiniLM-L6-v2",
        temp_dir: Optional[str] = None
    ):
        self.temp_dir = Path(temp_dir or settings.AUDIO_TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle Whisper model - either pre-loaded or load from name
        if whisper_model is None:
            self.whisper_model = None
        elif isinstance(whisper_model, str):
            logger.info(f"Loading Whisper model: {whisper_model}")
            self.whisper_model = whisper.load_model(whisper_model)
        else:
            logger.info("Using pre-loaded Whisper model")
            self.whisper_model = whisper_model
        
        # Handle embedding model - either pre-loaded or load from name
        if embed_model is None:
            self.embed_model = None
        elif isinstance(embed_model, str):
            logger.info(f"Loading embedding model: {embed_model}")
            self.embed_model = SentenceTransformer(embed_model)
        else:
            logger.info("Using pre-loaded embedding model")
            self.embed_model = embed_model
    
    def extract_audio(self, video_path: Path, audio_path: Optional[Path] = None) -> Path:
        """
        Extract audio stream from video using FFmpeg
        
        Args:
            video_path: Path to video file
            audio_path: Optional output path for audio
        
        Returns:
            Path to extracted audio file
        """
        if audio_path is None:
            video_id = video_path.stem
            audio_path = self.temp_dir / f"{video_id}.wav"
        
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        
        try:
            # Extract audio as WAV (16kHz mono for Whisper)
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(audio_path),
                acodec='pcm_s16le',
                ac=1,  # mono
                ar='16000'  # 16kHz
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Audio extracted to {audio_path}")
            return audio_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
    
    def transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper ASR
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Whisper transcription result with segments
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Transcribe with word-level timestamps
        result = self.whisper_model.transcribe(
            str(audio_path),
            word_timestamps=True,
            verbose=False
        )
        
        logger.info(f"Transcription complete. Found {len(result['segments'])} segments")
        return result
    
    def create_audio_events(self, transcription: Dict[str, Any]) -> List[AudioEvent]:
        """
        Create AudioEvent objects from Whisper transcription
        
        Args:
            transcription: Whisper transcription result
        
        Returns:
            List of AudioEvent objects with timestamps
        """
        audio_events = []
        
        for segment in transcription['segments']:
            event = AudioEvent(
                timestamp=segment['start'],
                text=segment['text'].strip(),
                confidence=segment.get('no_speech_prob', None),
                metadata={
                    'start': segment['start'],
                    'end': segment['end'],
                    'words': segment.get('words', [])
                }
            )
            audio_events.append(event)
        
        logger.info(f"Created {len(audio_events)} audio events")
        return audio_events
    
    def generate_embeddings(self, audio_events: List[AudioEvent]) -> np.ndarray:
        """
        Generate embeddings for audio transcript segments
        
        Args:
            audio_events: List of AudioEvent objects
        
        Returns:
            Numpy array of embeddings (n_segments, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(audio_events)} audio segments")
        
        # Extract text from events
        texts = [event.text for event in audio_events]
        
        # Generate embeddings
        embeddings = self.embed_model.encode(
            texts,
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def process(self, video_path: Path, timeline: MasterTimeline) -> tuple[List[AudioEvent], np.ndarray]:
        """
        Complete audio pipeline processing
        
        Args:
            video_path: Path to video file
            timeline: Master timeline to populate with audio events
        
        Returns:
            tuple: (audio_events, embeddings)
        """
        logger.info("Starting audio pipeline processing")
        
        # Step 1: Extract audio
        audio_path = self.extract_audio(video_path)
        
        # Step 2: Transcribe
        transcription = self.transcribe_audio(audio_path)
        
        # Step 3: Create events
        audio_events = self.create_audio_events(transcription)
        
        # Step 4: Add events to timeline
        for event in audio_events:
            timeline.add_audio_event(event)
        
        # Step 5: Generate embeddings
        embeddings = self.generate_embeddings(audio_events)
        
        logger.info("Audio pipeline processing complete")
        
        return audio_events, embeddings
    
    def get_transcript_text(self, audio_events: List[AudioEvent]) -> str:
        """Get full transcript as text"""
        return " ".join(event.text for event in audio_events)
    
    def get_transcript_with_timestamps(self, audio_events: List[AudioEvent]) -> str:
        """Get transcript with timestamps"""
        lines = []
        for event in audio_events:
            timestamp = f"[{event.timestamp:.2f}s]"
            lines.append(f"{timestamp} {event.text}")
        return "\n".join(lines)


# Convenience function
def process_audio(video_path: Path, timeline: MasterTimeline, **kwargs) -> tuple[List[AudioEvent], np.ndarray]:
    """
    Convenience function for audio processing
    
    Usage:
        audio_events, embeddings = process_audio(video_path, timeline)
    """
    pipeline = AudioPipeline(**kwargs)
    return pipeline.process(video_path, timeline)