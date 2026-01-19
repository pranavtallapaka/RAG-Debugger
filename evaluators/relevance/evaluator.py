"""
Relevance Evaluator

Evaluates how relevant retrieved documents are to a given query.
"""

from typing import TypedDict, Optional
import numpy as np

from utils.embeddings import get_embeddings


class ScoredChunk(TypedDict):
    """A chunk with its relevance score."""
    chunk: str
    score: float


class RelevanceEvaluator:
    """
    Evaluates the relevance of retrieved documents to queries.
    
    Uses cosine similarity between query and document embeddings
    to compute relevance scores. Includes a missing-context heuristic
    to detect potential retrieval failures.
    """
    
    def __init__(self, missing_context_threshold: float = 0.3):
        """
        Initialize the relevance evaluator.
        
        Args:
            missing_context_threshold: Maximum similarity threshold below which
                a warning is issued for potential retrieval failure (default: 0.3).
                Range: [0, 1]. Lower values = stricter (more warnings).
        
        Raises:
            ValueError: If threshold is not in valid range [0, 1].
        """
        if not 0.0 <= missing_context_threshold <= 1.0:
            raise ValueError(
                f"missing_context_threshold must be between 0 and 1, "
                f"got {missing_context_threshold}"
            )
        self.missing_context_threshold = missing_context_threshold
    
    def evaluate(
        self, 
        query: str, 
        chunks: list[str]
    ) -> list[ScoredChunk]:
        """
        Evaluate relevance of chunks to the query using cosine similarity.
        
        Computes embeddings for the query and all chunks, then calculates
        cosine similarity scores. Returns chunks sorted by relevance score
        in descending order.
        
        Args:
            query: The user's query string
            chunks: List of retrieved chunk texts to evaluate
            
        Returns:
            List of ScoredChunk dictionaries, sorted by score (descending).
            Each ScoredChunk contains:
                - chunk: The original chunk text
                - score: Relevance score (cosine similarity, range [-1, 1])
            
        Example:
            >>> evaluator = RelevanceEvaluator()
            >>> chunks = ["Python is a language", "Java is also a language"]
            >>> results = evaluator.evaluate("What is Python?", chunks)
            >>> print(results[0]["score"])  # Highest relevance score
        """
        if not chunks:
            return []
        
        # Get embeddings for query and chunks (already normalized)
        query_embedding = get_embeddings([query], normalize=True)[0]
        chunk_embeddings = get_embeddings(chunks, normalize=True)
        
        # Compute cosine similarity (dot product since embeddings are normalized)
        # Shape: (n_chunks,)
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Create list of scored chunks
        scored_chunks: list[ScoredChunk] = [
            {"chunk": chunk, "score": float(score)}
            for chunk, score in zip(chunks, similarities)
        ]
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_chunks
    
    def check_missing_context(
        self,
        query: str,
        chunks: list[str],
        similarities: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Heuristic to detect potential retrieval failures (missing context).
        
        If the maximum similarity between the query and any retrieved chunk
        is below the threshold, this suggests the retrieval system may have
        failed to find relevant content. Returns a warning message if detected.
        
        Args:
            query: The user's query string
            chunks: List of retrieved chunk texts to evaluate
            similarities: Optional pre-computed similarity scores (from evaluate()).
                         If provided, embeddings won't be recomputed, improving
                         efficiency when called after evaluate().
            
        Returns:
            Warning message string if missing context is detected, None otherwise.
            The message explains that retrieval likely failed and suggests
            checking the retrieval system configuration.
            
        Example:
            >>> evaluator = RelevanceEvaluator(missing_context_threshold=0.3)
            >>> chunks = ["Unrelated text about cooking"]
            >>> warning = evaluator.check_missing_context("Python programming", chunks)
            >>> if warning:
            ...     print(warning)  # Warning about low similarity
        """
        if not chunks:
            return (
                "⚠️ Missing Context Warning: No chunks retrieved. "
                "The retrieval system returned no results. Check your retrieval "
                "configuration and ensure documents are properly indexed."
            )
        
        # Use pre-computed similarities if available, otherwise compute them
        if similarities is None:
            # Get embeddings and compute similarities
            query_embedding = get_embeddings([query], normalize=True)[0]
            chunk_embeddings = get_embeddings(chunks, normalize=True)
            similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Find maximum similarity
        max_similarity = float(np.max(similarities))
        
        # Check if below threshold
        if max_similarity < self.missing_context_threshold:
            return (
                f"⚠️ Missing Context Warning: Maximum query-chunk similarity "
                f"({max_similarity:.3f}) is below threshold "
                f"({self.missing_context_threshold:.3f}). This suggests the "
                f"retrieval system may have failed to find relevant content. "
                f"Consider: (1) Checking retrieval parameters (top_k, similarity "
                f"threshold), (2) Verifying document indexing, (3) Reviewing "
                f"query formulation or expansion."
            )
        
        return None

