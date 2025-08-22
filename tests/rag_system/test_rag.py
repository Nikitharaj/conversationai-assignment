"""
Tests for the RAG class.
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

import pytest
from unittest.mock import patch, MagicMock

from src.rag_system.rag import RAG


class TestRAG(unittest.TestCase):
    """Test cases for the RAG class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "model"
        self.model_dir.mkdir(exist_ok=True)

        # Create test documents
        self.test_docs = [
            MagicMock(page_content="The revenue was $10 million in 2023."),
            MagicMock(page_content="The profit margin was 15% last year."),
        ]

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch("src.rag_system.rag.HuggingFaceEmbeddings")
    @patch("src.rag_system.rag.FAISS")
    @patch("src.rag_system.rag.HuggingFacePipeline")
    def test_initialization(self, mock_hf_pipeline, mock_faiss, mock_hf_embeddings):
        """Test initialization of RAG."""
        # Mock components
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        mock_pipeline = MagicMock()
        mock_hf_pipeline.return_value = mock_pipeline

        # Force langchain_available to be True for this test
        with patch("src.rag_system.rag.langchain_available", True):
            rag = RAG(
                embedding_model_name="all-MiniLM-L6-v2",
                llm_model_name="distilgpt2",
                retrieval_method="hybrid",
                top_k=3,
            )

            self.assertEqual(rag.embedding_model_name, "all-MiniLM-L6-v2")
            self.assertEqual(rag.llm_model_name, "distilgpt2")
            self.assertEqual(rag.retrieval_method, "hybrid")
            self.assertEqual(rag.top_k, 3)
            self.assertIsNone(rag.vector_store)
            self.assertIsNone(rag.retriever)
            self.assertIsNone(rag.chain)

    @patch("src.rag_system.rag.HuggingFaceEmbeddings")
    @patch("src.rag_system.rag.FAISS")
    @patch("src.rag_system.rag.RecursiveCharacterTextSplitter")
    def test_build_indexes(self, mock_splitter, mock_faiss, mock_hf_embeddings):
        """Test building indexes."""
        # Mock components
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance

        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance
        mock_splitter_instance.create_documents.return_value = self.test_docs

        # Force langchain_available to be True for this test
        with patch("src.rag_system.rag.langchain_available", True):
            rag = RAG(
                embedding_model_name="all-MiniLM-L6-v2", llm_model_name="distilgpt2"
            )

            # Build indexes
            chunks = [
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
            rag.build_indexes(chunks)

            mock_faiss.from_documents.assert_called_once()
            self.assertIsNotNone(rag.vector_store)

    @patch("src.rag_system.rag.HuggingFaceEmbeddings")
    @patch("src.rag_system.rag.FAISS")
    @patch("src.rag_system.rag.HuggingFacePipeline")
    @patch("src.rag_system.rag.LLMChain")
    def test_process_query(
        self, mock_chain, mock_hf_pipeline, mock_faiss, mock_hf_embeddings
    ):
        """Test processing a query."""
        # Mock components
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance
        mock_faiss_instance.as_retriever.return_value = MagicMock()

        mock_pipeline = MagicMock()
        mock_hf_pipeline.return_value = mock_pipeline

        mock_chain_instance = MagicMock()
        mock_chain.return_value = mock_chain_instance
        mock_chain_instance.run.return_value = "The revenue was $10 million."

        # Force langchain_available to be True for this test
        with patch("src.rag_system.rag.langchain_available", True):
            rag = RAG(
                embedding_model_name="all-MiniLM-L6-v2", llm_model_name="distilgpt2"
            )

            # Set up vector store and chain
            rag.vector_store = mock_faiss_instance
            rag.retriever = mock_faiss_instance.as_retriever()
            rag.chain = mock_chain_instance

            # Process query
            result = rag.process_query("What was the revenue?")
            self.assertIsInstance(result, dict)
            self.assertEqual(result["answer"], "The revenue was $10 million.")
            self.assertIn("confidence", result)
            self.assertIn("response_time", result)
            self.assertIn("retrieved_chunks", result)

            mock_chain_instance.run.assert_called_once()

    @patch("src.rag_system.rag.HuggingFaceEmbeddings")
    @patch("src.rag_system.rag.FAISS")
    def test_save_load(self, mock_faiss, mock_hf_embeddings):
        """Test saving and loading the model."""
        # Mock components
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings

        mock_faiss_instance = MagicMock()
        mock_faiss.from_documents.return_value = mock_faiss_instance
        mock_faiss.load_local.return_value = mock_faiss_instance

        # Force langchain_available to be True for this test
        with patch("src.rag_system.rag.langchain_available", True):
            rag = RAG(
                embedding_model_name="all-MiniLM-L6-v2", llm_model_name="distilgpt2"
            )

            # Set up vector store
            rag.vector_store = mock_faiss_instance

            # Save model
            rag.save(self.model_dir)
            mock_faiss_instance.save_local.assert_called_once()

            # Load model
            rag.load(self.model_dir)
            mock_faiss.load_local.assert_called_once()

    def test_fallback_functionality(self):
        """Test fallback functionality when LangChain is not available."""
        # Force langchain_available to be False for this test
        with patch("src.rag_system.rag.langchain_available", False):
            rag = RAG(
                embedding_model_name="all-MiniLM-L6-v2", llm_model_name="distilgpt2"
            )

            # Process query
            result = rag.process_query("What was the revenue?")
            self.assertIsInstance(result, dict)
            # Just check that we get a response with the expected structure
            self.assertIn("answer", result)
            self.assertIsInstance(result["answer"], str)


if __name__ == "__main__":
    unittest.main()
