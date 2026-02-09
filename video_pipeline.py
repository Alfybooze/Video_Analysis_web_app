"""
Video Pipeline
- Extracts frames from video at specified rate
- Creates timestamped frames
- Generates image embeddings using CLIP/SigLIP
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
import open_clip
from loguru import logger
from tqdm import tqdm

from settings import settings
from timeline import MasterTimeline, FrameEvent


class VideoPipeline:
    """
    Video Processing Pipeline
    
    FFmpeg → Frame Sampling (1fps / scenes) → Timestamped Frames → Image Embeddings (CLIP / SigLIP)
    """
    
    def __init__(
        self,
        frame_rate: Optional[float] = None,
        vision_model: Optional[str] = None,
        vision_pretrained: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        self.frame_rate = frame_rate or settings.FRAME_SAMPLE_RATE
        self.vision_model_name = vision_model or settings.VISION_MODEL
        self.vision_pretrained = vision_pretrained or settings.VISION_PRETRAINED
        self.output_dir = Path(output_dir or settings.VIDEO_TEMP_DIR) / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CLIP model
        logger.info(f"Loading vision model: {self.vision_model_name}")
        self.device = settings.DEVICE
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.vision_model_name,
            pretrained=self.vision_pretrained,
            device=self.device
        )
        self.model.eval()
        
        logger.info(f"Vision model loaded on device: {self.device}")
    
    def extract_frames(
        self,
        video_path: Path,
        video_id: str,
        save_frames: bool = True
    ) -> List[Tuple[float, np.ndarray, Optional[Path]]]:
        """
        Extract frames from video at specified frame rate
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            save_frames: Whether to save frames to disk
        
        Returns:
            List of (timestamp, frame_array, frame_path) tuples
        """
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
        
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
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
                pbar.update(1)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def create_frame_events(
        self,
        frames: List[Tuple[float, np.ndarray, Optional[Path]]],
        video_id: str
    ) -> List[FrameEvent]:
        """
        Create FrameEvent objects from extracted frames
        
        Args:
            frames: List of (timestamp, frame_array, frame_path) tuples
            video_id: Unique video identifier
        
        Returns:
            List of FrameEvent objects
        """
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
    
    def generate_embeddings(
        self,
        frames: List[Tuple[float, np.ndarray, Optional[Path]]]
    ) -> np.ndarray:
        """
        Generate CLIP embeddings for frames
        
        Args:
            frames: List of (timestamp, frame_array, frame_path) tuples
        
        Returns:
            Numpy array of embeddings (n_frames, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(frames)} frames")
        
        embeddings = []
        
        # Process in batches
        batch_size = settings.BATCH_SIZE
        
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size), desc="Encoding frames"):
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
                
                # Convert to numpy
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings_np)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        logger.info(f"Generated embeddings with shape {all_embeddings.shape}")
        return all_embeddings
    
    def process(
        self,
        video_path: Path,
        video_id: str,
        timeline: MasterTimeline,
        save_frames: bool = True
    ) -> Tuple[List[FrameEvent], np.ndarray]:
        """
        Complete video pipeline processing
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            timeline: Master timeline to populate with frame events
            save_frames: Whether to save frames to disk
        
        Returns:
            tuple: (frame_events, embeddings)
        """
        logger.info("Starting video pipeline processing")
        
        # Step 1: Extract frames
        frames = self.extract_frames(video_path, video_id, save_frames)
        
        # Step 2: Create frame events
        frame_events = self.create_frame_events(frames, video_id)
        
        # Step 3: Add events to timeline
        for event in frame_events:
            timeline.add_frame_event(event)
        
        # Step 4: Generate embeddings
        embeddings = self.generate_embeddings(frames)
        
        logger.info("Video pipeline processing complete")
        
        return frame_events, embeddings
    
    def detect_scene_changes(
        self,
        video_path: Path,
        threshold: float = 30.0
    ) -> List[float]:
        """
        Detect scene changes in video (alternative to fixed frame rate)
        
        Args:
            video_path: Path to video file
            threshold: Scene change threshold
        
        Returns:
            List of timestamps where scenes change
        """
        logger.info(f"Detecting scene changes in {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        scene_timestamps = []
        prev_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    timestamp = frame_count / video_fps
                    scene_timestamps.append(timestamp)
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Detected {len(scene_timestamps)} scene changes")
        return scene_timestamps


# Convenience function
def process_video(
    video_path: Path,
    video_id: str,
    timeline: MasterTimeline,
    **kwargs
) -> Tuple[List[FrameEvent], np.ndarray]:
    """
    Convenience function for video processing
    
    Usage:
        frame_events, embeddings = process_video(video_path, video_id, timeline)
    """
    pipeline = VideoPipeline(**kwargs)
    return pipeline.process(video_path, video_id, timeline, **kwargs)