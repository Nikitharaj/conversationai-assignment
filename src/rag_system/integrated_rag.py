"""
Integrated LangChain-based RAG system.

This module combines all LangChain components into a unified RAG system.
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple

# Import project components
from .document_chunker import DocumentChunker
from .embedding_manager import EmbeddingManager
from .answer_generator import AnswerGenerator

# Try to import LangChain
try:
    from langchain_core.documents import Document

    langchain_available = True
except ImportError:
    warnings.warn("LangChain not available. Install with 'pip install langchain-core'")
    langchain_available = False


class IntegratedRAG:
    """Integrated LangChain-based RAG system."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = "distilgpt2",
        chunk_sizes: List[int] = [100, 400],
        chunk_overlap: int = 50,
        retrieval_method: str = "hybrid",
        top_k: int = 5,
    ):
        """
        Initialize the integrated LangChain RAG system.

        Args:
            embedding_model: Name of the embedding model
            llm_model: Name of the language model (None to disable)
            chunk_sizes: List of token/character sizes for chunking
            chunk_overlap: Number of tokens/characters to overlap between chunks
            retrieval_method: Retrieval method ("dense", "sparse", or "hybrid")
            top_k: Number of chunks to retrieve
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        self.retrieval_method = retrieval_method
        self.top_k = top_k

        # Initialize components
        self.chunker = DocumentChunker(
            chunk_sizes=chunk_sizes, chunk_overlap=chunk_overlap
        )
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.answer_generator = AnswerGenerator(model_name=llm_model)

        if langchain_available:
            print("LangChain components initialized successfully")
        else:
            print("Using fallback components due to LangChain unavailability")

        # Track if the system is initialized with documents
        self.is_initialized = False

    def initialize_from_documents(
        self, documents_dir: Union[str, Path], chunk_size_idx: int = 1
    ):
        """
        Initialize the RAG system from documents.

        Args:
            documents_dir: Directory containing processed document files
            chunk_size_idx: Index of chunk size to use (0 for small, 1 for large)
        """
        documents_dir = Path(documents_dir)

        # Check if the directory exists
        if not documents_dir.exists():
            raise FileNotFoundError(f"Directory not found: {documents_dir}")

        # Get all text files
        text_files = list(documents_dir.glob("*.txt"))

        if not text_files:
            raise ValueError(f"No text files found in {documents_dir}")

        # Chunk all documents
        all_chunks = []
        for file_path in text_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Use the specified chunk size
            chunk_size = self.chunk_sizes[chunk_size_idx]
            chunks = self.chunker.chunk_document(text, chunk_size)

            # Add document metadata to each chunk
            for chunk in chunks:
                chunk["document"] = file_path.stem
                all_chunks.append(chunk)

        # Build indexes
        self.embedding_manager.build_indexes(all_chunks)

        self.is_initialized = True
        print(
            f"Initialized from {len(text_files)} documents with {len(all_chunks)} chunks"
        )

    def initialize_from_chunks(self, chunks_file: Union[str, Path]):
        """
        Initialize the RAG system from pre-chunked documents.

        Args:
            chunks_file: Path to a JSON file containing document chunks
        """
        chunks_file = Path(chunks_file)

        # Check if the file exists
        if not chunks_file.exists():
            raise FileNotFoundError(f"File not found: {chunks_file}")

        # Load chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Build indexes
        self.embedding_manager.build_indexes(chunks)

        self.is_initialized = True
        print(f"Initialized from {len(chunks)} pre-chunked documents")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and generate an answer.

        Args:
            query: User query

        Returns:
            Dictionary containing the answer and metadata
        """
        if not self.is_initialized:
            raise ValueError(
                "RAG system not initialized. Call initialize_from_documents or initialize_from_chunks first."
            )

        # Apply input-side guardrails
        filtered_query, is_filtered = self.answer_generator.filter_query(query)

        if is_filtered:
            return {
                "query": query,
                "answer": filtered_query,
                "confidence": 1.0,
                "response_time": 0.0,
                "is_filtered": True,
                "retrieved_chunks": [],
            }

        # Retrieve relevant chunks
        if self.retrieval_method == "dense":
            retrieved_chunks = self.embedding_manager.dense_search(
                query, top_k=self.top_k
            )
        elif self.retrieval_method == "sparse":
            retrieved_chunks = self.embedding_manager.sparse_search(
                query, top_k=self.top_k
            )
        else:  # hybrid
            retrieved_chunks = self.embedding_manager.hybrid_search(
                query, top_k=self.top_k
            )

        # Generate answer
        answer, confidence, response_time = self.answer_generator.generate_answer(
            query, retrieved_chunks
        )

        # Apply output-side guardrails
        answer, is_hallucination = self.answer_generator.apply_guardrails(
            query, answer, retrieved_chunks
        )

        # Adjust confidence if hallucination detected
        if is_hallucination:
            confidence *= 0.5

        return {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "response_time": response_time,
            "is_filtered": False,
            "is_hallucination": is_hallucination,
            "retrieved_chunks": retrieved_chunks,
        }

    def save(self, output_dir: Union[str, Path]):
        """
        Save the RAG system to disk.

        Args:
            output_dir: Directory to save the system
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "chunk_sizes": self.chunk_sizes,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_method": self.retrieval_method,
            "top_k": self.top_k,
        }

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save embedding indexes
        embedding_dir = output_dir / "embeddings"
        self.embedding_manager.save_indexes(embedding_dir)

        # Save LLM configuration
        llm_dir = output_dir / "llm"
        llm_dir.mkdir(parents=True, exist_ok=True)
        self.answer_generator.save(llm_dir)

        print(f"RAG system saved to {output_dir}")

    def load(self, input_dir: Union[str, Path]):
        """
        Load the RAG system from disk.

        Args:
            input_dir: Directory containing the saved system
        """
        input_dir = Path(input_dir)

        # Load configuration
        with open(input_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Update configuration
        self.embedding_model = config["embedding_model"]
        self.llm_model = config["llm_model"]
        self.chunk_sizes = config["chunk_sizes"]
        self.chunk_overlap = config["chunk_overlap"]
        self.retrieval_method = config["retrieval_method"]
        self.top_k = config["top_k"]

        # Reinitialize components with new configuration
        self.chunker = DocumentChunker(
            chunk_sizes=self.chunk_sizes, chunk_overlap=self.chunk_overlap
        )

        # Load embedding indexes
        embedding_dir = input_dir / "embeddings"
        self.embedding_manager.load_indexes(embedding_dir)

        # Load LLM configuration
        llm_dir = input_dir / "llm"
        self.answer_generator.load(llm_dir)

        self.is_initialized = True
        print(f"RAG system loaded from {input_dir}")


if __name__ == "__main__":
    # Example usage
    rag_system = IntegratedRAG(
        embedding_model="all-MiniLM-L6-v2",
        llm_model=None,  # Disable LLM to avoid segmentation faults
        chunk_sizes=[100, 400],
        chunk_overlap=50,
        retrieval_method="sparse",
        top_k=3,
    )

    # Example chunks
    chunks = [
        {
            "text": "The company reported revenue of $10.5 million for Q2 2023.",
            "document": "financial_report.pdf",
        },
        {
            "text": "This represents a 15% increase from the same period last year.",
            "document": "financial_report.pdf",
        },
        {
            "text": "Operating expenses were $8.2 million, resulting in a profit margin of 21.9%.",
            "document": "financial_report.pdf",
        },
    ]

    # Build indexes
    rag_system.embedding_manager.build_indexes(chunks)
    rag_system.is_initialized = True

    # Process a query
    result = rag_system.process_query("What was the revenue in Q2 2023?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Response time: {result['response_time']:.3f}s")
