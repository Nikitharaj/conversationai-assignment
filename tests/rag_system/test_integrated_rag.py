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


if __name__ == "__main__":
    unittest.main()
