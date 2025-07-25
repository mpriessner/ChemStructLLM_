"""
Vector store service for embedding and retrieving molecular data.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """Service for storing and retrieving vector embeddings of molecular data."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector store."""
        self.model = SentenceTransformer(model_name)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_item(
        self,
        key: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an item to the vector store."""
        # Generate embedding
        embedding = self.model.encode([text])[0]
        
        # Store embedding and metadata
        self.embeddings[key] = embedding
        if metadata:
            self.metadata[key] = metadata
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar items in the vector store."""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        results = []
        for key, embedding in self.embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= threshold:
                result = {
                    'key': key,
                    'similarity': float(similarity),
                    'metadata': self.metadata.get(key, {})
                }
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_item(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item by key."""
        if key in self.embeddings:
            return {
                'embedding': self.embeddings[key],
                'metadata': self.metadata.get(key, {})
            }
        return None
    
    def remove_item(self, key: str) -> bool:
        """Remove an item from the vector store."""
        if key in self.embeddings:
            del self.embeddings[key]
            if key in self.metadata:
                del self.metadata[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from the vector store."""
        self.embeddings.clear()
        self.metadata.clear()
