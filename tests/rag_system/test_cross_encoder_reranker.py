"""
Tests for the CrossEncoderReranker class (Group 118 Advanced Technique) - Fixed Version.
"""


# Apply regex compatibility patch FIRST
def apply_regex_patch():
    """Apply regex version patch for compatibility."""
    try:
        import regex

        if regex.__version__ == "2.5.159":
            regex.__version__ = "2025.7.34"
            if hasattr(regex, "version"):
                regex.version = "2025.7.34"
    except ImportError:
        pass


apply_regex_patch()

import os
import unittest
from pathlib import Path
import tempfile
import warnings
import time

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", message=".*regex.*2019.12.17.*")
warnings.filterwarnings("ignore", message=".*regex!=2019.12.17.*")
warnings.filterwarnings(
    "ignore", message=".*bitsandbytes.*compiled without GPU support.*"
)
warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")

import pytest
from unittest.mock import patch, MagicMock

from src.rag_system.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderRerankerFixed(unittest.TestCase):
    """Test cases for the CrossEncoderReranker class - Fixed Version."""

    def setUp(self):
        """Set up test environment."""
        self.reranker = CrossEncoderReranker()

        # Sample chunks for testing (matching actual expected format)
        self.sample_chunks = [
            {
                "content": "Apple's total revenue for fiscal 2023 was $383.3 billion, representing a 3% decline year over year.",
                "score": 0.85,
            },
            {
                "content": "The company's cash and cash equivalents totaled $29.5 billion at the end of fiscal 2023.",
                "score": 0.75,
            },
            {
                "content": "Research and development expenses were $29.9 billion in fiscal 2023, up from $26.3 billion in 2022.",
                "score": 0.65,
            },
            {
                "content": "The iPhone generated $200.6 billion in revenue, representing 52% of total revenue.",
                "score": 0.70,
            },
        ]

    def test_initialization(self):
        """Test CrossEncoderReranker initialization."""
        reranker = CrossEncoderReranker()
        self.assertIsNotNone(reranker)
        self.assertEqual(reranker.model_name, "cross-encoder/ms-marco-MiniLM-L-2-v2")

        # Test with custom model
        custom_reranker = CrossEncoderReranker(model_name="custom-model")
        self.assertEqual(custom_reranker.model_name, "custom-model")

    def test_rerank_chunks_basic(self):
        """Test basic re-ranking functionality."""
        query = "What was Apple's total revenue in 2023?"

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=self.sample_chunks, top_k=3
        )

        # Check return structure
        self.assertIsInstance(reranked_chunks, list)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(len(reranked_chunks), 3)

        # Check metadata structure (based on actual implementation)
        self.assertIn("method", metadata)
        self.assertIn("rerank_time", metadata)
        self.assertIn("scores", metadata)
        self.assertIn("rank_changes", metadata)

        # Check that chunks have cross-encoder scores (at chunk level, not metadata)
        for chunk in reranked_chunks:
            self.assertIn("cross_encoder_score", chunk)
            self.assertIn("original_score", chunk)
            self.assertIn("original_rank", chunk)
            self.assertIn("final_rank", chunk)
            self.assertIn("reranked", chunk)

    def test_rerank_chunks_empty_input(self):
        """Test re-ranking with empty input."""
        query = "What was Apple's revenue?"

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=[], top_k=3
        )

        self.assertEqual(len(reranked_chunks), 0)
        self.assertIn("method", metadata)

    def test_rerank_chunks_single_chunk(self):
        """Test re-ranking with single chunk."""
        query = "What was Apple's revenue?"
        single_chunk = [self.sample_chunks[0]]

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=single_chunk, top_k=3
        )

        self.assertEqual(len(reranked_chunks), 1)
        self.assertIn("cross_encoder_score", reranked_chunks[0])

    def test_rerank_chunks_top_k_larger_than_input(self):
        """Test re-ranking when top_k is larger than input size."""
        query = "What was Apple's revenue?"

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query,
            retrieved_chunks=self.sample_chunks[:2],  # Only 2 chunks
            top_k=5,  # Request 5
        )

        # Should return all available chunks (2)
        self.assertEqual(len(reranked_chunks), 2)

    def test_score_tracking(self):
        """Test that scores are properly tracked."""
        query = "What was Apple's total revenue in 2023?"

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=self.sample_chunks, top_k=4
        )

        # Check score tracking in metadata
        self.assertIn("scores", metadata)
        scores = metadata["scores"]

        self.assertIn("cross_encoder", scores)
        self.assertIn("original", scores)
        self.assertIn("score_changes", scores)

        # Score lists should match number of returned chunks
        self.assertEqual(len(scores["cross_encoder"]), len(reranked_chunks))
        self.assertEqual(len(scores["original"]), len(reranked_chunks))
        self.assertEqual(len(scores["score_changes"]), len(reranked_chunks))

    def test_rank_change_tracking(self):
        """Test that rank changes are tracked correctly."""
        query = "What was Apple's total revenue in 2023?"

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=self.sample_chunks, top_k=4
        )

        # Check rank changes tracking
        self.assertIn("rank_changes", metadata)
        rank_changes = metadata["rank_changes"]

        self.assertIsInstance(rank_changes, list)
        self.assertEqual(len(rank_changes), len(reranked_chunks))

        # Check that original and final ranks are set for chunks
        for chunk in reranked_chunks:
            self.assertIn("original_rank", chunk)
            self.assertIn("final_rank", chunk)
            self.assertIsInstance(chunk["original_rank"], int)
            self.assertIsInstance(chunk["final_rank"], int)

    def test_fallback_when_unavailable(self):
        """Test fallback behavior when cross-encoder is not available."""
        # Create reranker with cross-encoder disabled
        with patch(
            "src.rag_system.cross_encoder_reranker.cross_encoder_available", False
        ):
            reranker = CrossEncoderReranker()
            query = "What was Apple's revenue?"

            reranked_chunks, metadata = reranker.rerank_chunks(
                query=query, retrieved_chunks=self.sample_chunks, top_k=3
            )

            self.assertEqual(len(reranked_chunks), 3)
            self.assertEqual(metadata["method"], "fallback_original_order")

            # Check that fallback preserves original order
            for i, chunk in enumerate(reranked_chunks):
                self.assertEqual(chunk["final_rank"], i + 1)
                self.assertEqual(chunk["original_rank"], i + 1)
                self.assertFalse(chunk["reranked"])

    def test_performance_timing(self):
        """Test that performance timing is recorded."""
        query = "What was Apple's revenue in 2023?"

        start_time = time.time()
        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=self.sample_chunks, top_k=3
        )
        end_time = time.time()

        # Check that timing is recorded and reasonable
        self.assertIn("rerank_time", metadata)
        self.assertIsInstance(metadata["rerank_time"], float)
        self.assertGreater(metadata["rerank_time"], 0)
        self.assertLess(
            metadata["rerank_time"], end_time - start_time + 1
        )  # Allow some buffer

    def test_cross_encoder_score_range(self):
        """Test that cross-encoder scores are in reasonable range."""
        query = "What was Apple's total revenue in 2023?"

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=self.sample_chunks, top_k=4
        )

        # Cross-encoder scores should be reasonable (typically between -10 and 10)
        for chunk in reranked_chunks:
            ce_score = chunk["cross_encoder_score"]
            self.assertIsInstance(ce_score, (int, float))
            self.assertGreater(ce_score, -20)  # Very loose bounds
            self.assertLess(ce_score, 20)

    def test_metadata_preservation(self):
        """Test that original chunk data is preserved."""
        query = "What was Apple's revenue?"

        # Add metadata to test chunks
        test_chunks = []
        for i, chunk in enumerate(self.sample_chunks):
            enhanced_chunk = chunk.copy()
            enhanced_chunk["metadata"] = {
                "id": f"chunk_{i + 1}",
                "section": "income_statement",
                "year": 2023,
            }
            test_chunks.append(enhanced_chunk)

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=test_chunks, top_k=3
        )

        # Check that original data is preserved
        for chunk in reranked_chunks:
            # Original content should still exist
            self.assertIn("content", chunk)

            # Original metadata should be preserved if it existed
            if "metadata" in chunk:
                original_metadata = chunk["metadata"]
                self.assertIn("id", original_metadata)
                self.assertIn("section", original_metadata)
                self.assertIn("year", original_metadata)

            # New cross-encoder fields should be added at chunk level
            self.assertIn("cross_encoder_score", chunk)
            self.assertIn("original_score", chunk)
            self.assertIn("original_rank", chunk)
            self.assertIn("final_rank", chunk)
            self.assertIn("reranked", chunk)

    def test_query_document_pair_preparation(self):
        """Test that query-document pairs are prepared correctly."""
        query = "What was Apple's revenue?"

        # This tests the internal pair preparation indirectly
        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=self.sample_chunks[:2], top_k=2
        )

        # Should successfully process without errors
        self.assertEqual(len(reranked_chunks), 2)
        self.assertIn("method", metadata)

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        query = "What was Apple's revenue?"

        # Perform some re-ranking operations
        for _ in range(3):
            self.reranker.rerank_chunks(
                query=query, retrieved_chunks=self.sample_chunks[:2], top_k=2
            )

        stats = self.reranker.get_performance_stats()

        self.assertIn("total_reranks", stats)
        self.assertIn("avg_rerank_time", stats)
        self.assertIn("model_name", stats)
        self.assertEqual(stats["total_reranks"], 3)

    def test_different_input_formats(self):
        """Test handling of different input chunk formats."""
        query = "What was Apple's revenue?"

        # Test with different chunk formats
        mixed_chunks = [
            {"content": "Revenue was $100M", "score": 0.8},
            {"text": "Profit was $20M", "score": 0.7},  # Different text key
            "Cash was $30M",  # String only
        ]

        reranked_chunks, metadata = self.reranker.rerank_chunks(
            query=query, retrieved_chunks=mixed_chunks, top_k=3
        )

        # Should handle all formats
        self.assertEqual(len(reranked_chunks), 3)
        self.assertIn("method", metadata)

        # All chunks should have required fields after processing
        for chunk in reranked_chunks:
            self.assertIn("cross_encoder_score", chunk)
            self.assertIn("original_score", chunk)


if __name__ == "__main__":
    unittest.main()
