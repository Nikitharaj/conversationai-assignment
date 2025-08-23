"""
Cross-Encoder Re-ranking for RAG System (Group 118 Advanced Technique).

This module implements cross-encoder re-ranking to improve retrieval quality
by jointly scoring query-document pairs for better relevance ranking.
"""

import warnings
from typing import List, Dict, Any, Tuple, Optional
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    import torch

    cross_encoder_available = True
    logger.info(" Cross-encoder components available")
except ImportError as e:
    warnings.warn(
        f"Cross-encoder dependencies not available: {e}. Install with 'pip install sentence-transformers torch'"
    )
    cross_encoder_available = False


class CrossEncoderReranker:
    """
    Cross-Encoder Re-ranker for improving RAG retrieval quality.

    This implements the Group 118 advanced technique for RAG systems.
    Uses a cross-encoder model to jointly score (query, document) pairs
    for better relevance ranking than individual embeddings.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        Initialize the Cross-Encoder Re-ranker.

        Args:
            model_name: Name of the cross-encoder model (MS-MARCO style)
            max_length: Maximum sequence length for input
            batch_size: Batch size for processing multiple pairs
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        self.is_initialized = False

        # Performance tracking
        self.rerank_times = []
        self.score_cache = {}

        if cross_encoder_available:
            self._initialize_model()
        else:
            logger.warning(
                "Cross-encoder not available. Re-ranking will use fallback scoring."
            )

    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device="cpu",  # Use CPU to avoid memory issues
            )
            self.is_initialized = True
            logger.info(" Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.is_initialized = False

    def rerank_chunks(
        self, query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = 4
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Re-rank retrieved chunks using cross-encoder scoring.

        Args:
            query: The user query
            retrieved_chunks: List of chunks from initial retrieval
            top_k: Number of top chunks to return after re-ranking

        Returns:
            Tuple of (re-ranked chunks, rerank_metadata)
        """
        start_time = time.time()

        if not retrieved_chunks:
            return [], {"rerank_time": 0, "scores": [], "method": "no_chunks"}

        # Store original scores and order
        original_scores = []
        original_order = []

        for i, chunk_data in enumerate(retrieved_chunks):
            if isinstance(chunk_data, dict) and "score" in chunk_data:
                original_scores.append(chunk_data["score"])
            else:
                original_scores.append(1.0 - (i * 0.1))  # Fallback scoring
            original_order.append(i)

        if not self.is_initialized or not cross_encoder_available:
            # Fallback: return original ranking with enhanced metadata
            rerank_time = time.time() - start_time
            return self._fallback_rerank(
                retrieved_chunks, original_scores, top_k, rerank_time
            )

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = []
            chunk_texts = []

            for chunk_data in retrieved_chunks:
                # Extract text from chunk data
                if isinstance(chunk_data, dict):
                    if "chunk" in chunk_data and isinstance(chunk_data["chunk"], dict):
                        text = chunk_data["chunk"].get("text", "")
                    elif "text" in chunk_data:
                        text = chunk_data["text"]
                    else:
                        text = str(chunk_data)
                else:
                    text = str(chunk_data)

                # Truncate text if too long
                if len(text) > 400:  # Leave room for query in max_length
                    text = text[:400] + "..."

                chunk_texts.append(text)
                pairs.append([query, text])

            # Get cross-encoder scores
            logger.info(f"Re-ranking {len(pairs)} chunks with cross-encoder")
            cross_encoder_scores = self.model.predict(pairs)

            # Convert to list if numpy array
            if hasattr(cross_encoder_scores, "tolist"):
                cross_encoder_scores = cross_encoder_scores.tolist()

            # Combine with original data and sort by cross-encoder score
            scored_chunks = []
            for i, (chunk_data, ce_score) in enumerate(
                zip(retrieved_chunks, cross_encoder_scores)
            ):
                enhanced_chunk = (
                    chunk_data.copy()
                    if isinstance(chunk_data, dict)
                    else {"text": str(chunk_data)}
                )
                enhanced_chunk.update(
                    {
                        "cross_encoder_score": float(ce_score),
                        "original_score": original_scores[i],
                        "original_rank": i + 1,
                        "reranked": True,
                    }
                )
                scored_chunks.append(enhanced_chunk)

            # Sort by cross-encoder score (descending)
            scored_chunks.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

            # Take top-k
            reranked_chunks = scored_chunks[:top_k]

            # Add final ranks
            for i, chunk in enumerate(reranked_chunks):
                chunk["final_rank"] = i + 1

            rerank_time = time.time() - start_time
            self.rerank_times.append(rerank_time)

            # Prepare metadata
            metadata = {
                "rerank_time": rerank_time,
                "method": "cross_encoder",
                "model_name": self.model_name,
                "original_count": len(retrieved_chunks),
                "reranked_count": len(reranked_chunks),
                "scores": {
                    "cross_encoder": [
                        chunk["cross_encoder_score"] for chunk in reranked_chunks
                    ],
                    "original": [chunk["original_score"] for chunk in reranked_chunks],
                    "score_changes": [
                        chunk["cross_encoder_score"] - chunk["original_score"]
                        for chunk in reranked_chunks
                    ],
                },
                "rank_changes": [
                    chunk["original_rank"] - chunk["final_rank"]
                    for chunk in reranked_chunks
                ],
            }

            logger.info(
                f" Re-ranking completed in {rerank_time:.3f}s, returned top-{len(reranked_chunks)} chunks"
            )

            return reranked_chunks, metadata

        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            # Fallback to original ranking
            rerank_time = time.time() - start_time
            return self._fallback_rerank(
                retrieved_chunks, original_scores, top_k, rerank_time
            )

    def _fallback_rerank(
        self,
        chunks: List[Dict[str, Any]],
        scores: List[float],
        top_k: int,
        rerank_time: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fallback re-ranking when cross-encoder is not available.

        Uses original scores with some enhancements.
        """
        # Enhance chunks with metadata
        enhanced_chunks = []
        for i, (chunk_data, score) in enumerate(zip(chunks, scores)):
            enhanced_chunk = (
                chunk_data.copy()
                if isinstance(chunk_data, dict)
                else {"text": str(chunk_data)}
            )
            enhanced_chunk.update(
                {
                    "cross_encoder_score": score,  # Use original score as fallback
                    "original_score": score,
                    "original_rank": i + 1,
                    "final_rank": i + 1,
                    "reranked": False,
                }
            )
            enhanced_chunks.append(enhanced_chunk)

        # Take top-k
        result_chunks = enhanced_chunks[:top_k]

        metadata = {
            "rerank_time": rerank_time,
            "method": "fallback_original_order",
            "model_name": "none",
            "original_count": len(chunks),
            "reranked_count": len(result_chunks),
            "scores": {
                "cross_encoder": [
                    chunk["cross_encoder_score"] for chunk in result_chunks
                ],
                "original": [chunk["original_score"] for chunk in result_chunks],
                "score_changes": [0.0] * len(result_chunks),  # No changes in fallback
            },
            "rank_changes": [0] * len(result_chunks),  # No rank changes in fallback
        }

        return result_chunks, metadata

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the re-ranker."""
        if not self.rerank_times:
            return {"message": "No re-ranking operations performed yet"}

        return {
            "total_reranks": len(self.rerank_times),
            "avg_rerank_time": sum(self.rerank_times) / len(self.rerank_times),
            "min_rerank_time": min(self.rerank_times),
            "max_rerank_time": max(self.rerank_times),
            "total_rerank_time": sum(self.rerank_times),
            "model_name": self.model_name,
            "is_initialized": self.is_initialized,
        }

    def clear_cache(self):
        """Clear the score cache to free memory."""
        self.score_cache.clear()
        logger.info("Score cache cleared")


# Utility function for easy integration
def create_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
) -> CrossEncoderReranker:
    """
    Factory function to create a cross-encoder re-ranker.

    Args:
        model_name: Name of the cross-encoder model

    Returns:
        CrossEncoderReranker instance
    """
    return CrossEncoderReranker(model_name=model_name)


if __name__ == "__main__":
    # Example usage and testing
    reranker = create_cross_encoder_reranker()

    # Test data
    query = "What was the revenue in 2023?"
    chunks = [
        {
            "text": "Revenue for 2023 was $1,250 million, up from $1,162 million in 2022.",
            "score": 0.8,
        },
        {
            "text": "The company's expenses increased significantly in 2023.",
            "score": 0.7,
        },
        {
            "text": "Total revenue was $1,250 million for fiscal year 2023.",
            "score": 0.9,
        },
        {"text": "Cash flow improved in the fourth quarter of 2023.", "score": 0.6},
    ]

    # Test re-ranking
    reranked, metadata = reranker.rerank_chunks(query, chunks, top_k=3)

    print("Re-ranking Results:")
    print(f"Method: {metadata['method']}")
    print(f"Time: {metadata['rerank_time']:.3f}s")
    print("\nTop chunks:")
    for i, chunk in enumerate(reranked):
        print(
            f"{i + 1}. Score: {chunk['cross_encoder_score']:.3f} | {chunk['text'][:100]}..."
        )
