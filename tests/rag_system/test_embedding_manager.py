"""
Tests for the EmbeddingManager class.
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

import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", message=".*regex.*2019.12.17.*")
warnings.filterwarnings("ignore", message=".*regex!=2019.12.17.*")
warnings.filterwarnings("ignore", message=".*bitsandbytes.*compiled without GPU support.*")
warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")


import os
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np

import pytest
from unittest.mock import patch, MagicMock

from src.rag_system.embedding_manager import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    """Test cases for the EmbeddingManager class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.indexes_dir = Path(self.temp_dir) / "indexes"
        self.indexes_dir.mkdir(exist_ok=True)

        # Create test chunks
        self.test_chunks = [
            {"text": "This is the first test chunk.", "start_idx": 0, "end_idx": 28},
            {"text": "This is the second test chunk.", "start_idx": 29, "end_idx": 58},
            {"text": "This is the third test chunk.", "start_idx": 59, "end_idx": 87},
        ]

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch("src.rag_system.embedding_manager.HuggingFaceEmbeddings")
    def test_initialization(self, mock_hf_embeddings):
        """Test initialization of EmbeddingManager."""
        # Mock HuggingFaceEmbeddings
        mock_instance = MagicMock()
        mock_hf_embeddings.return_value = mock_instance

        # Force langchain_available to be True for this test
        with patch("src.rag_system.embedding_manager.langchain_available", True):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            self.assertEqual(manager.model_name, "all-MiniLM-L6-v2")
            self.assertIsNone(manager.dense_retriever)
            self.assertIsNone(manager.sparse_retriever)
            self.assertEqual(manager.chunks, [])

    @patch("src.rag_system.embedding_manager.HuggingFaceEmbeddings")
    @patch("src.rag_system.embedding_manager.FAISS")
    def test_build_indexes(self, mock_faiss, mock_hf_embeddings):
        """Test building indexes."""
        # Mock HuggingFaceEmbeddings
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # Force langchain_available to be True for this test
        with patch("src.rag_system.embedding_manager.langchain_available", True):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            manager.build_indexes(self.test_chunks)

            self.assertEqual(manager.chunks, self.test_chunks)
            mock_faiss.from_documents.assert_called_once()

    @patch("src.rag_system.embedding_manager.HuggingFaceEmbeddings")
    @patch("src.rag_system.embedding_manager.FAISS")
    @patch("src.rag_system.embedding_manager.BM25Retriever")
    def test_dense_search(self, mock_bm25, mock_faiss, mock_hf_embeddings):
        """Test dense search."""
        # Mock HuggingFaceEmbeddings
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance
        mock_faiss_instance.similarity_search_with_score.return_value = [
            (
                MagicMock(
                    page_content="This is the first test chunk.",
                    metadata={"start_idx": 0, "end_idx": 28},
                ),
                0.9,
            ),
            (
                MagicMock(
                    page_content="This is the second test chunk.",
                    metadata={"start_idx": 29, "end_idx": 58},
                ),
                0.8,
            ),
        ]

        # Force langchain_available to be True for this test
        with patch("src.rag_system.embedding_manager.langchain_available", True):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            manager.build_indexes(self.test_chunks)

            results = manager.dense_search("test query", top_k=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["score"], 0.9)
            self.assertEqual(results[0]["method"], "dense")
            self.assertEqual(
                results[0]["chunk"]["text"], "This is the first test chunk."
            )

    @patch("src.rag_system.embedding_manager.HuggingFaceEmbeddings")
    @patch("src.rag_system.embedding_manager.BM25Retriever")
    def test_sparse_search(self, mock_bm25, mock_hf_embeddings):
        """Test sparse search."""
        # Mock HuggingFaceEmbeddings
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        # Mock BM25Retriever
        mock_bm25_instance = MagicMock()
        mock_bm25.return_value = mock_bm25_instance
        mock_bm25_instance.get_relevant_documents.return_value = [
            MagicMock(
                page_content="This is the first test chunk.",
                metadata={"start_idx": 0, "end_idx": 28, "score": 0.85},
            ),
            MagicMock(
                page_content="This is the third test chunk.",
                metadata={"start_idx": 59, "end_idx": 87, "score": 0.75},
            ),
        ]

        # Force langchain_available to be True for this test
        with patch("src.rag_system.embedding_manager.langchain_available", True):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            manager.build_indexes(self.test_chunks)

            results = manager.sparse_search("test query", top_k=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["method"], "sparse")
            self.assertEqual(
                results[0]["chunk"]["text"], "This is the first test chunk."
            )

    @patch("src.rag_system.embedding_manager.HuggingFaceEmbeddings")
    @patch("src.rag_system.embedding_manager.FAISS")
    @patch("src.rag_system.embedding_manager.BM25Retriever")
    @patch("src.rag_system.embedding_manager.EnsembleRetriever")
    def test_hybrid_search(
        self, mock_ensemble, mock_bm25, mock_faiss, mock_hf_embeddings
    ):
        """Test hybrid search."""
        # Mock HuggingFaceEmbeddings
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # Mock BM25Retriever
        mock_bm25_instance = MagicMock()
        mock_bm25.return_value = mock_bm25_instance

        # Mock EnsembleRetriever
        mock_ensemble_instance = MagicMock()
        mock_ensemble.return_value = mock_ensemble_instance
        mock_ensemble_instance.get_relevant_documents.return_value = [
            MagicMock(
                page_content="This is the first test chunk.",
                metadata={"start_idx": 0, "end_idx": 28, "score": 0.95},
            ),
            MagicMock(
                page_content="This is the second test chunk.",
                metadata={"start_idx": 29, "end_idx": 58, "score": 0.85},
            ),
        ]

        # Force langchain_available to be True for this test
        with patch("src.rag_system.embedding_manager.langchain_available", True):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            manager.build_indexes(self.test_chunks)

            results = manager.hybrid_search("test query", top_k=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["method"], "hybrid")
            self.assertEqual(
                results[0]["chunk"]["text"], "This is the first test chunk."
            )

    @patch("src.rag_system.embedding_manager.HuggingFaceEmbeddings")
    @patch("src.rag_system.embedding_manager.FAISS")
    def test_save_load_indexes(self, mock_faiss, mock_hf_embeddings):
        """Test saving and loading indexes."""
        # Mock HuggingFaceEmbeddings
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        # Force langchain_available to be True for this test
        with patch("src.rag_system.embedding_manager.langchain_available", True):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            manager.build_indexes(self.test_chunks)

            # Save indexes
            manager.save_indexes(self.indexes_dir)

            # Check that files were created
            self.assertTrue((self.indexes_dir / "chunks.json").exists())

            # Load indexes
            new_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            new_manager.load_indexes(self.indexes_dir)

            # Check that chunks were loaded
            self.assertEqual(len(new_manager.chunks), len(self.test_chunks))

    def test_fallback_functionality(self):
        """Test fallback functionality when LangChain is not available."""
        # Force langchain_available to be False for this test
        with patch("src.rag_system.embedding_manager.langchain_available", False):
            manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
            manager.build_indexes(self.test_chunks)

            # Test search methods
            dense_results = manager.dense_search("test query")
            sparse_results = manager.sparse_search("test query")
            hybrid_results = manager.hybrid_search("test query")

            # All should return empty lists in fallback mode
            self.assertEqual(dense_results, [])
            self.assertEqual(sparse_results, [])
            self.assertEqual(hybrid_results, [])


if __name__ == "__main__":
    unittest.main()
