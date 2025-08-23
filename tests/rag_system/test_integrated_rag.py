"""
Tests for the IntegratedRAG class.
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
import shutil
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", message=".*regex.*2019.12.17.*")
warnings.filterwarnings("ignore", message=".*regex!=2019.12.17.*")
warnings.filterwarnings(
    "ignore", message=".*bitsandbytes.*compiled without GPU support.*"
)
warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")

import pytest
from unittest.mock import patch, MagicMock

from src.rag_system.integrated_rag import IntegratedRAG


class TestIntegratedRAG(unittest.TestCase):
    """Test cases for the IntegratedRAG class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "model"
        self.model_dir.mkdir(exist_ok=True)

        # Create test chunks
        self.test_chunks = [
            {
                "text": "The revenue was $10 million in 2023.",
                "start_idx": 0,
                "end_idx": 35,
            },
            {
                "text": "The profit margin was 15% last year.",
                "start_idx": 36,
                "end_idx": 72,
            },
        ]

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch("src.rag_system.document_chunker.DocumentChunker")
    @patch("src.rag_system.embedding_manager.EmbeddingManager")
    @patch("src.rag_system.answer_generator.AnswerGenerator")
    def test_initialization(
        self, mock_answer_generator, mock_embedding_manager, mock_document_chunker
    ):
        """Test initialization of IntegratedRAG."""
        # Mock components
        mock_chunker = MagicMock()
        mock_document_chunker.return_value = mock_chunker

        mock_embeddings = MagicMock()
        mock_embedding_manager.return_value = mock_embeddings

        mock_generator = MagicMock()
        mock_answer_generator.return_value = mock_generator

        # Force langchain_available to be True for this test
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                chunk_sizes=[100, 400],
                chunk_overlap=50,
                retrieval_method="hybrid",
                top_k=3,
            )

            self.assertEqual(rag.embedding_model, "all-MiniLM-L6-v2")
            self.assertEqual(rag.llm_model, "distilgpt2")
            self.assertEqual(rag.chunk_sizes, [100, 400])
            self.assertEqual(rag.chunk_overlap, 50)
            self.assertEqual(rag.retrieval_method, "hybrid")
            self.assertEqual(rag.top_k, 3)
            self.assertFalse(rag.is_initialized)

            # Check that components were initialized
            mock_document_chunker.assert_called_once()
            mock_embedding_manager.assert_called_once()
            mock_answer_generator.assert_called_once()

    @patch("src.rag_system.document_chunker.DocumentChunker")
    @patch("src.rag_system.embedding_manager.EmbeddingManager")
    @patch("src.rag_system.answer_generator.AnswerGenerator")
    def test_process_document(
        self, mock_answer_generator, mock_embedding_manager, mock_document_chunker
    ):
        """Test processing a document."""
        # Mock components
        mock_chunker = MagicMock()
        mock_document_chunker.return_value = mock_chunker
        mock_chunker.chunk_document.return_value = self.test_chunks

        mock_embeddings = MagicMock()
        mock_embedding_manager.return_value = mock_embeddings

        mock_generator = MagicMock()
        mock_answer_generator.return_value = mock_generator

        # Force langchain_available to be True for this test
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2", llm_model="distilgpt2"
            )

            # Process document
            test_file = Path(self.temp_dir) / "test_doc.txt"
            with open(test_file, "w") as f:
                f.write("This is a test document.")

            result = rag.process_document(test_file)
            self.assertEqual(result, "Document processed successfully.")
            mock_chunker.chunk_document.assert_called_once()
            mock_embeddings.build_indexes.assert_called_once()
            self.assertTrue(rag.is_initialized)

    @patch("src.rag_system.document_chunker.DocumentChunker")
    @patch("src.rag_system.embedding_manager.EmbeddingManager")
    @patch("src.rag_system.answer_generator.AnswerGenerator")
    def test_process_query(
        self, mock_answer_generator, mock_embedding_manager, mock_document_chunker
    ):
        """Test processing a query."""
        # Mock components
        mock_chunker = MagicMock()
        mock_document_chunker.return_value = mock_chunker

        mock_embeddings = MagicMock()
        mock_embedding_manager.return_value = mock_embeddings
        retrieved_chunks = [
            {
                "chunk": {"text": "The revenue was $10 million."},
                "score": 0.9,
                "method": "hybrid",
            }
        ]
        mock_embeddings.hybrid_search.return_value = retrieved_chunks
        mock_embeddings.dense_search.return_value = retrieved_chunks
        mock_embeddings.sparse_search.return_value = retrieved_chunks

        mock_generator = MagicMock()
        mock_answer_generator.return_value = mock_generator
        mock_generator.generate_answer.return_value = (
            "The revenue was $10 million.",
            0.9,
            0.1,
        )

        # Force langchain_available to be True for this test
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                retrieval_method="hybrid",
            )
            rag.is_initialized = True

            # Process query
            result = rag.process_query("What was the revenue?")
            self.assertIsInstance(result, dict)
            self.assertEqual(result["answer"], "The revenue was $10 million.")
            self.assertEqual(result["confidence"], 0.9)
            self.assertIn("response_time", result)
            self.assertEqual(result["retrieved_chunks"], retrieved_chunks)

            # Test with different retrieval methods
            rag.retrieval_method = "dense"
            rag.process_query("What was the revenue?")
            mock_embeddings.dense_search.assert_called_once()

            rag.retrieval_method = "sparse"
            rag.process_query("What was the revenue?")
            mock_embeddings.sparse_search.assert_called_once()

    @patch("src.rag_system.document_chunker.DocumentChunker")
    @patch("src.rag_system.embedding_manager.EmbeddingManager")
    @patch("src.rag_system.answer_generator.AnswerGenerator")
    def test_save_load(
        self, mock_answer_generator, mock_embedding_manager, mock_document_chunker
    ):
        """Test saving and loading the model."""
        # Mock components
        mock_chunker = MagicMock()
        mock_document_chunker.return_value = mock_chunker

        mock_embeddings = MagicMock()
        mock_embedding_manager.return_value = mock_embeddings

        mock_generator = MagicMock()
        mock_answer_generator.return_value = mock_generator

        # Force langchain_available to be True for this test
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2", llm_model="distilgpt2"
            )
            rag.is_initialized = True

            # Save model
            rag.save(self.model_dir)
            mock_embeddings.save_indexes.assert_called_once()

            # Load model
            rag.load(self.model_dir)
            mock_embeddings.load_indexes.assert_called_once()

    def test_fallback_functionality(self):
        """Test fallback functionality when LangChain is not available."""
        # Force langchain_available to be False for this test
        with patch("src.rag_system.integrated_rag.langchain_available", False):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2", llm_model="distilgpt2"
            )

            # Initialize with sample chunks
            sample_chunks = [{"text": "Sample revenue data", "document": "test.txt"}]
            rag.initialize_from_chunks(sample_chunks)

            # Process query
            result = rag.process_query("What was the revenue?")
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertIn("confidence", result)

    def test_cross_encoder_integration(self):
        """Test cross-encoder re-ranking integration."""
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                use_cross_encoder=True,
            )

            # Check that cross-encoder is initialized
            self.assertIsNotNone(rag.cross_encoder_reranker)
            self.assertTrue(rag.use_cross_encoder)

    @patch("src.rag_system.document_chunker.DocumentChunker")
    @patch("src.rag_system.embedding_manager.EmbeddingManager")
    @patch("src.rag_system.answer_generator.AnswerGenerator")
    def test_process_query_with_cross_encoder(
        self, mock_answer_generator, mock_embedding_manager, mock_document_chunker
    ):
        """Test query processing with cross-encoder re-ranking."""
        # Mock components
        mock_chunker = MagicMock()
        mock_document_chunker.return_value = mock_chunker

        mock_embeddings = MagicMock()
        mock_embedding_manager.return_value = mock_embeddings

        # Mock retrieved chunks
        mock_chunks = [
            {
                "content": "Revenue was $100M in 2023",
                "metadata": {"id": "chunk1", "score": 0.8},
                "score": 0.8,
            },
            {
                "content": "Profit was $20M in 2023",
                "metadata": {"id": "chunk2", "score": 0.7},
                "score": 0.7,
            },
        ]
        mock_embeddings.hybrid_search.return_value = mock_chunks

        mock_generator = MagicMock()
        mock_generator.generate_answer.return_value = {
            "answer": "Revenue was $100M",
            "confidence": 0.9,
        }
        mock_answer_generator.return_value = mock_generator

        # Force langchain_available to be True for this test
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                use_cross_encoder=True,
            )
            rag.is_initialized = True

            # Mock cross-encoder re-ranking
            reranked_chunks = [
                {
                    "content": "Revenue was $100M in 2023",
                    "metadata": {
                        "id": "chunk1",
                        "score": 0.8,
                        "cross_encoder_score": 0.95,
                        "reranked": True,
                    },
                    "score": 0.8,
                }
            ]
            rerank_metadata = {
                "method": "cross_encoder",
                "rerank_time": 0.1,
                "score_changes": {"improved": 1, "degraded": 0, "unchanged": 0},
            }

            with patch.object(
                rag.cross_encoder_reranker,
                "rerank_chunks",
                return_value=(reranked_chunks, rerank_metadata),
            ):
                result = rag.process_query("What was the revenue?")

                # Check that cross-encoder metadata is included
                self.assertIn("rerank_metadata", result)
                self.assertIn("cross_encoder_used", result)
                self.assertTrue(result["cross_encoder_used"])
                self.assertEqual(result["rerank_metadata"]["method"], "cross_encoder")

    def test_cross_encoder_disabled(self):
        """Test behavior when cross-encoder is disabled."""
        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                use_cross_encoder=False,
            )

            # Cross-encoder should not be used
            self.assertFalse(rag.use_cross_encoder)

    @patch("src.rag_system.document_chunker.DocumentChunker")
    @patch("src.rag_system.embedding_manager.EmbeddingManager")
    @patch("src.rag_system.answer_generator.AnswerGenerator")
    def test_retrieval_expansion_for_reranking(
        self, mock_answer_generator, mock_embedding_manager, mock_document_chunker
    ):
        """Test that retrieval is expanded when cross-encoder is enabled."""
        # Mock components
        mock_chunker = MagicMock()
        mock_document_chunker.return_value = mock_chunker

        mock_embeddings = MagicMock()
        mock_embedding_manager.return_value = mock_embeddings
        mock_embeddings.hybrid_search.return_value = []

        mock_generator = MagicMock()
        mock_generator.generate_answer.return_value = {
            "answer": "No answer",
            "confidence": 0.1,
        }
        mock_answer_generator.return_value = mock_generator

        with patch("src.rag_system.integrated_rag.langchain_available", True):
            rag = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="distilgpt2",
                top_k=4,
                use_cross_encoder=True,
            )
            rag.is_initialized = True

            # Mock cross-encoder
            with patch.object(
                rag.cross_encoder_reranker,
                "rerank_chunks",
                return_value=([], {"method": "cross_encoder"}),
            ):
                rag.process_query("Test query")

                # Check that hybrid_search was called with expanded top_k
                # Should be max(top_k * 3, 12) = max(4 * 3, 12) = 12
                mock_embeddings.hybrid_search.assert_called_once()
                call_args = mock_embeddings.hybrid_search.call_args
                self.assertEqual(call_args[1]["top_k"], 12)


if __name__ == "__main__":
    unittest.main()
