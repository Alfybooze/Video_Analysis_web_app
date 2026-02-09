"""
Master Timeline - Single Source of Truth for Video Analytics
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta
import json


@dataclass
class TimeRange:
    """Represents a time range in the video"""
    start_seconds: float
    end_seconds: float
    
    def __post_init__(self):
        if self.start_seconds > self.end_seconds:
            raise ValueError("start_seconds must be <= end_seconds")
    
    def overlaps(self, other: 'TimeRange') -> bool:
        """Check if this time range overlaps with another"""
        return not (self.end_seconds < other.start_seconds or self.start_seconds > other.end_seconds)
    
    def contains(self, timestamp: float) -> bool:
        """Check if a timestamp is within this range"""
        return self.start_seconds <= timestamp <= self.end_seconds
    
    def duration(self) -> float:
        """Get duration of this time range"""
        return self.end_seconds - self.start_seconds
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration": self.duration()
        }


@dataclass
class TimestampedEvent:
    """Base class for timestamped events"""
    timestamp: float
    event_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "metadata": self.metadata
        }


@dataclass
class AudioEvent(TimestampedEvent):
    """Audio-related event (transcript segment)"""
    text: str = ""
    confidence: Optional[float] = None
    
    def __post_init__(self):
        self.event_type = "audio"
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "text": self.text,
            "confidence": self.confidence
        })
        return data


@dataclass
class FrameEvent(TimestampedEvent):
    """Video frame event"""
    frame_index: int = 0
    frame_path: Optional[str] = None
    embedding_id: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = "frame"
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "embedding_id": self.embedding_id
        })
        return data


class MasterTimeline:
    """
    Master Timeline - Single source of truth for all temporal data
    
    Time is the backbone — audio and vision never fight each other.
    All events are aligned to the same temporal axis.
    """
    
    def __init__(self, video_id: str, duration_seconds: float, metadata: Optional[Dict] = None):
        self.video_id = video_id
        self.duration_seconds = duration_seconds
        self.metadata = metadata or {}
        
        # Separate event streams
        self.audio_events: List[AudioEvent] = []
        self.frame_events: List[FrameEvent] = []
        self.custom_events: List[TimestampedEvent] = []
        
        # Index for fast temporal lookups
        self._audio_index: Dict[float, AudioEvent] = {}
        self._frame_index: Dict[float, FrameEvent] = {}
    
    def add_audio_event(self, event: AudioEvent) -> None:
        """Add an audio event to the timeline"""
        if event.timestamp > self.duration_seconds:
            raise ValueError(f"Event timestamp {event.timestamp} exceeds video duration {self.duration_seconds}")
        
        self.audio_events.append(event)
        self._audio_index[event.timestamp] = event
    
    def add_frame_event(self, event: FrameEvent) -> None:
        """Add a frame event to the timeline"""
        if event.timestamp > self.duration_seconds:
            raise ValueError(f"Event timestamp {event.timestamp} exceeds video duration {self.duration_seconds}")
        
        self.frame_events.append(event)
        self._frame_index[event.timestamp] = event
    
    def add_custom_event(self, event: TimestampedEvent) -> None:
        """Add a custom event to the timeline"""
        if event.timestamp > self.duration_seconds:
            raise ValueError(f"Event timestamp {event.timestamp} exceeds video duration {self.duration_seconds}")
        
        self.custom_events.append(event)
    
    def get_events_in_range(self, time_range: TimeRange) -> Dict[str, List[TimestampedEvent]]:
        """
        Get all events within a time range
        Returns dict with 'audio', 'frames', and 'custom' keys
        """
        return {
            "audio": [e for e in self.audio_events if time_range.contains(e.timestamp)],
            "frames": [e for e in self.frame_events if time_range.contains(e.timestamp)],
            "custom": [e for e in self.custom_events if time_range.contains(e.timestamp)]
        }
    
    def get_nearest_frame(self, timestamp: float) -> Optional[FrameEvent]:
        """Get the frame nearest to a given timestamp"""
        if not self.frame_events:
            return None
        
        # Find frame with minimum time difference
        return min(self.frame_events, key=lambda f: abs(f.timestamp - timestamp))
    
    def get_frames_in_range(self, time_range: TimeRange, max_frames: Optional[int] = None) -> List[FrameEvent]:
        """Get frames within a time range, optionally limited"""
        frames = [f for f in self.frame_events if time_range.contains(f.timestamp)]
        
        if max_frames and len(frames) > max_frames:
            # Sample evenly
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]
        
        return frames
    
    def get_audio_transcript(self, time_range: TimeRange) -> str:
        """Get concatenated transcript for a time range"""
        audio_events = [e for e in self.audio_events if time_range.contains(e.timestamp)]
        return " ".join(e.text for e in sorted(audio_events, key=lambda x: x.timestamp))
    
    def align_audio_to_frame(self, timestamp: float, window_seconds: float = 2.0) -> Tuple[Optional[FrameEvent], List[AudioEvent]]:
        """
        Align audio events to a frame timestamp
        Returns the nearest frame and audio within a time window
        """
        frame = self.get_nearest_frame(timestamp)
        time_range = TimeRange(
            start_seconds=max(0, timestamp - window_seconds),
            end_seconds=min(self.duration_seconds, timestamp + window_seconds)
        )
        audio_events = [e for e in self.audio_events if time_range.contains(e.timestamp)]
        
        return frame, audio_events
    
    def to_dict(self) -> Dict[str, Any]:
        """Export timeline to dictionary"""
        return {
            "video_id": self.video_id,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
            "audio_events": [e.to_dict() for e in self.audio_events],
            "frame_events": [e.to_dict() for e in self.frame_events],
            "custom_events": [e.to_dict() for e in self.custom_events]
        }
    
    def save_json(self, filepath: str) -> None:
        """Save timeline to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MasterTimeline':
        """Load timeline from dictionary"""
        timeline = cls(
            video_id=data["video_id"],
            duration_seconds=data["duration_seconds"],
            metadata=data.get("metadata", {})
        )
        
        # Reconstruct events
        for audio_data in data.get("audio_events", []):
            event = AudioEvent(
                timestamp=audio_data["timestamp"],
                event_type=audio_data["event_type"],
                text=audio_data["text"],
                confidence=audio_data.get("confidence"),
                metadata=audio_data.get("metadata", {})
            )
            timeline.add_audio_event(event)
        
        for frame_data in data.get("frame_events", []):
            event = FrameEvent(
                timestamp=frame_data["timestamp"],
                event_type=frame_data["event_type"],
                frame_index=frame_data["frame_index"],
                frame_path=frame_data.get("frame_path"),
                embedding_id=frame_data.get("embedding_id"),
                metadata=frame_data.get("metadata", {})
            )
            timeline.add_frame_event(event)
        
        return timeline
    
    @classmethod
    def load_json(cls, filepath: str) -> 'MasterTimeline':
        """Load timeline from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        return (f"MasterTimeline(video_id='{self.video_id}', "
                f"duration={self.duration_seconds}s, "
                f"audio_events={len(self.audio_events)}, "
                f"frame_events={len(self.frame_events)})")