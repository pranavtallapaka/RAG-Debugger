"""
Embedding utility using sentence-transformers.

Provides a lightweight interface for generating normalized embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Global model instance (lazy-loaded)
_model = None


def _get_model():
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def get_embeddings(texts: list[str], normalize: bool = True) -> np.ndarray:
    """
    Generate normalized embeddings for a list of text strings.
    
    Args:
        texts: List of text strings to embed
        normalize: Whether to L2-normalize the embeddings (default: True)
        
    Returns:
        numpy array of shape (n_texts, embedding_dim) with normalized embeddings.
        For empty input, returns shape (0, embedding_dim) for consistency.
        
    Example:
        >>> texts = ["Hello world", "How are you?"]
        >>> embeddings = get_embeddings(texts)
        >>> print(embeddings.shape)  # (2, 384)
    """
    if not texts:
        # Return empty array with correct shape: (0, embedding_dim)
        # This ensures consistent shape for downstream operations
        model = _get_model()
        embedding_dim = model.get_sentence_embedding_dimension()
        return np.array([]).reshape(0, embedding_dim)
    
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    if normalize:
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
    
    return embeddings

