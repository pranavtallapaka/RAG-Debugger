"""
Redundancy Evaluator

Detects redundant or duplicate information across retrieved documents.
"""

from typing import TypedDict
import numpy as np

from utils.embeddings import get_embeddings


class RedundantPair(TypedDict):
    """A pair of redundant chunks with their similarity score."""
    chunk1: str
    chunk2: str
    similarity: float


class RedundancyResult(TypedDict):
    """Result of redundancy evaluation."""
    flagged_pairs: list[RedundantPair]
    redundancy_ratio: float


class RedundancyEvaluator:
    """
    Evaluates redundancy in retrieved documents.
    
    Computes pairwise cosine similarity between all chunks and flags
    pairs that exceed a similarity threshold. Calculates an overall
    redundancy ratio based on the number of flagged pairs.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the redundancy evaluator.
        
        Args:
            similarity_threshold: Cosine similarity threshold above which
                chunks are considered redundant (default: 0.85).
                Range: [0, 1]. Higher values = stricter (less redundancy detected).
        
        Raises:
            ValueError: If threshold is not in valid range [0, 1].
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, "
                f"got {similarity_threshold}"
            )
        self.similarity_threshold = similarity_threshold
    
    def evaluate(self, chunks: list[str]) -> RedundancyResult:
        """
        Evaluate redundancy across chunks using pairwise cosine similarity.
        
        Computes embeddings for all chunks, then calculates pairwise cosine
        similarity. Flags pairs above the similarity threshold and computes
        an overall redundancy ratio.
        
        Args:
            chunks: List of retrieved chunk texts to evaluate
            
        Returns:
            RedundancyResult dictionary containing:
                - flagged_pairs: List of RedundantPair dictionaries for pairs
                  above the threshold, sorted by similarity (descending).
                  Each pair contains:
                    - chunk1: First chunk text
                    - chunk2: Second chunk text
                    - similarity: Cosine similarity score [0, 1]
                - redundancy_ratio: Ratio of flagged pairs to total possible
                  pairs. Range: [0, 1]. Higher = more redundancy.
                  
        Example:
            >>> evaluator = RedundancyEvaluator(similarity_threshold=0.8)
            >>> chunks = ["Python is great", "Python is great", "Java is good"]
            >>> result = evaluator.evaluate(chunks)
            >>> print(f"Redundancy ratio: {result['redundancy_ratio']:.2f}")
        """
        if len(chunks) < 2:
            return {
                "flagged_pairs": [],
                "redundancy_ratio": 0.0
            }
        
        # Get embeddings for all chunks (already normalized)
        chunk_embeddings = get_embeddings(chunks, normalize=True)
        
        # Compute pairwise cosine similarity matrix
        # Since embeddings are normalized, dot product = cosine similarity
        # Shape: (n_chunks, n_chunks)
        similarity_matrix = np.dot(chunk_embeddings, chunk_embeddings.T)
        
        # Find pairs above threshold (only upper triangle to avoid duplicates)
        flagged_pairs: list[RedundantPair] = []
        n_chunks = len(chunks)
        
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                similarity = float(similarity_matrix[i, j])
                if similarity >= self.similarity_threshold:
                    flagged_pairs.append({
                        "chunk1": chunks[i],
                        "chunk2": chunks[j],
                        "similarity": similarity
                    })
        
        # Sort by similarity descending
        flagged_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Compute redundancy ratio: flagged pairs / total possible pairs
        total_pairs = n_chunks * (n_chunks - 1) / 2
        redundancy_ratio = len(flagged_pairs) / total_pairs if total_pairs > 0 else 0.0
        
        return {
            "flagged_pairs": flagged_pairs,
            "redundancy_ratio": redundancy_ratio
        }

