"""
Coverage Evaluator

Assesses how well the retrieved content covers the query requirements.
"""

from typing import TypedDict
import numpy as np

from utils.embeddings import get_embeddings


class CoverageResult(TypedDict):
    """Result of coverage evaluation."""
    coverage_score: float
    average_similarity: float
    max_similarity: float
    min_similarity: float


class CoverageEvaluator:
    """
    Evaluates coverage of query requirements by retrieved documents.
    
    Uses embedding similarity to assess how well the retrieved chunks
    collectively cover the query. Computes aggregate statistics.
    """
    
    def __init__(self):
        """Initialize the coverage evaluator."""
        pass
    
    def evaluate(self, query: str, chunks: list[str]) -> CoverageResult:
        """
        Evaluate coverage of query by chunks.
        
        Computes similarity between query and all chunks, then calculates
        aggregate statistics to assess overall coverage quality.
        
        Args:
            query: The user's query string
            chunks: List of retrieved chunk texts to evaluate
            
        Returns:
            CoverageResult dictionary containing:
                - coverage_score: Average similarity across all chunks [0, 1]
                - average_similarity: Mean query-chunk similarity
                - max_similarity: Highest query-chunk similarity
                - min_similarity: Lowest query-chunk similarity
        """
        if not chunks:
            return {
                "coverage_score": 0.0,
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0
            }
        
        # Get embeddings and compute similarities
        query_embedding = get_embeddings([query], normalize=True)[0]
        chunk_embeddings = get_embeddings(chunks, normalize=True)
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Compute statistics
        avg_sim = float(np.mean(similarities))
        max_sim = float(np.max(similarities))
        min_sim = float(np.min(similarities))
        
        # Coverage score is the average similarity
        coverage_score = avg_sim
        
        return {
            "coverage_score": coverage_score,
            "average_similarity": avg_sim,
            "max_similarity": max_sim,
            "min_similarity": min_sim
        }

