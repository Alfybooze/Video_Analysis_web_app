import faiss
import numpy as np
import pickle
import os
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import torch
import logging
from settings import settings


def check_faiss_gpu_availability():
    """Check if FAISS GPU functionality is available."""
    try:
        # Try to access GPU-related functions
        hasattr(faiss, 'index_cpu_to_gpu')
        hasattr(faiss, 'StandardGpuResources')
        return True
    except AttributeError:
        return False


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the vector store."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the vector store."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with GPU support when available."""
    
    def __init__(self, dimension: int, use_gpu: bool = False):
        """Initialize FAISS vector store.
        
        Args:
            dimension: Vector dimension
            use_gpu: Whether to use GPU. If None, will auto-detect based on settings and availability.
        """
        self.dimension = dimension
        self.use_gpu = use_gpu if use_gpu is not None else (settings.DEVICE == "cuda")
        self.gpu_available = check_faiss_gpu_availability()
        
        # Override GPU usage if GPU functionality is not available
        if self.use_gpu and not self.gpu_available:
            logging.warning("GPU requested but FAISS GPU functionality not available. Falling back to CPU.")
            self.use_gpu = False
        
        # Create index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Move to GPU if requested and available
        if self.use_gpu and self.gpu_available:
            try:
                self.res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
                logging.info("Using GPU for FAISS index")
            except Exception as e:
                logging.warning(f"Failed to move FAISS index to GPU: {e}. Falling back to CPU.")
                self.use_gpu = False
        else:
            logging.info("Using CPU for FAISS index")
        
        self.metadata = []
        self.id_map = {}
        self.next_id = 0
    
    def add(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add vectors to the store."""
        if len(vectors) == 0:
            return []
        
        # Normalize vectors for cosine similarity
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        
        # Generate IDs and store metadata
        ids = []
        for i, vector in enumerate(vectors):
            vector_id = str(self.next_id)
            ids.append(vector_id)
            self.id_map[vector_id] = len(self.metadata)
            
            meta = metadata[i] if metadata and i < len(metadata) else {}
            meta['vector_id'] = vector_id
            meta['index_position'] = len(self.metadata)
            self.metadata.append(meta)
            
            self.next_id += 1
        
        return ids
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(distance)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save(self, path: str):
        """Save the vector store."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Move index to CPU if it's on GPU for saving
        if self.use_gpu and self.gpu_available:
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, path + ".index")
            except Exception as e:
                logging.warning(f"Failed to convert GPU index to CPU for saving: {e}")
                faiss.write_index(self.index, path + ".index")
        else:
            faiss.write_index(self.index, path + ".index")
        
        # Save metadata
        with open(path + ".meta", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_map': self.id_map,
                'next_id': self.next_id,
                'dimension': self.dimension,
                'use_gpu': self.use_gpu,
                'gpu_available': self.gpu_available
            }, f)
    
    def load(self, path: str):
        """Load the vector store."""
        # Load index
        self.index = faiss.read_index(path + ".index")
        self.dimension = self.index.d
        
        # Load metadata
        with open(path + ".meta", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.id_map = data['id_map']
            self.next_id = data['next_id']
            self.use_gpu = data.get('use_gpu', False)
            self.gpu_available = data.get('gpu_available', False)
        
        # Move to GPU if it was originally on GPU and GPU is available
        if self.use_gpu and self.gpu_available and check_faiss_gpu_availability():
            try:
                self.res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
                logging.info("Loaded FAISS index to GPU")
            except Exception as e:
                logging.warning(f"Failed to move loaded FAISS index to GPU: {e}. Keeping on CPU.")
                self.use_gpu = False


class MultimodalVectorStore:
    """Vector store that handles multiple modalities (text, image, audio, video)."""
    
    def __init__(self, text_dim: int = 384, image_dim: int = 512, audio_dim: int = 384, video_dim: int = 512):
        """Initialize multimodal vector store."""
        self.text_store = FAISSVectorStore(text_dim)
        self.image_store = FAISSVectorStore(image_dim)
        self.audio_store = FAISSVectorStore(audio_dim)
        self.video_store = FAISSVectorStore(video_dim)
        
        self.modalities = {
            'text': self.text_store,
            'image': self.image_store,
            'audio': self.audio_store,
            'video': self.video_store
        }
    
    def add(self, modality: str, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add vectors for a specific modality."""
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        return self.modalities[modality].add(vectors, metadata)
    
    def search(self, modality: str, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific modality."""
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        return self.modalities[modality].search(query_vector, k)
    
    def search_all(self, query_vector: np.ndarray, k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all modalities."""
        results = {}
        for modality, store in self.modalities.items():
            results[modality] = store.search(query_vector, k)
        return results
    
    def save(self, base_path: str):
        """Save all modality stores."""
        for modality, store in self.modalities.items():
            store.save(f"{base_path}_{modality}")
    
    def load(self, base_path: str):
        """Load all modality stores."""
        for modality, store in self.modalities.items():
            store.load(f"{base_path}_{modality}")