"""
Configuration module for Video Analytics System
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
        # API Keys
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Video Processing
    FRAME_SAMPLE_RATE: float = Field(default=1.0, description="Frames per second to extract")
    MAX_VIDEO_DURATION: int = Field(default=3600, description="Max video duration in seconds")
    VIDEO_TEMP_DIR: str = Field(default="./temp/videos")
    
    # Audio Processing
    WHISPER_MODEL: str = Field(default="base", description="Whisper model size: tiny, base, small, medium, large")
    AUDIO_TEMP_DIR: str = Field(default="./temp/audio")
    
    # Embeddings
    VISION_MODEL: str = Field(default="ViT-B-32", description="CLIP/SigLIP model")
    VISION_PRETRAINED: str = Field(default="openai")
    AUDIO_EMBED_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=512)
    
    # Vector Store
    VECTOR_STORE_TYPE: Literal["faiss", "milvus"] = Field(default="faiss")
    VECTOR_STORE_PATH: str = Field(default="./data/vector_stores")
    TOP_K_RESULTS: int = Field(default=10)
    
    # LLM Settings
    LLM_PROVIDER: Literal["gemini", "anthropic"] = Field(default="gemini")
    MAX_FRAMES_TO_LLM: int = Field(default=10, description="Max frames to send to LLM")
    LLM_MAX_TOKENS: int = Field(default=4096)
    LLM_TEMPERATURE: float = Field(default=0.7)
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./data/video_analytics.db")
    
    # Processing
    BATCH_SIZE: int = Field(default=32)
    NUM_WORKERS: int = Field(default=4)
    DEVICE: str = Field(default="cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    
    # Storage
    DATA_DIR: str = Field(default="./data")
    OUTPUT_DIR: str = Field(default="./outputs")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings