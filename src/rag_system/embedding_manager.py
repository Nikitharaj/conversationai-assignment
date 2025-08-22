"""
LangChain-based embedding manager for document retrieval.

This module replaces the custom EmbeddingManager with LangChain's embeddings and vector stores.
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple


# Define Document class for fallback when langchain is not available
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Try to import LangChain components
try:
    # Import langchain first
    import langchain

    # Import core components
    import langchain_core

    # Import community components
    import langchain_community

    # Embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Vector stores
    from langchain_community.vectorstores import FAISS, Chroma

    # Retrievers
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers.ensemble import EnsembleRetriever

    # Document processing
    from langchain_core.documents import Document

    langchain_available = True
    print("LangChain components initialized successfully in embedding_manager")
except ImportError as e:
    warnings.warn(
        f"LangChain not available: {e}. Install with 'pip install langchain langchain-community'"
    )
    langchain_available = False


class EmbeddingManager:
    """LangChain-based embedding manager for document retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.retriever = None
        self.chunks = []
        self.is_initialized = False

        # Initialize embeddings if LangChain is available
        if langchain_available:
            self._initialize_embeddings()
        else:
            warnings.warn("LangChain not available. Using fallback retrieval methods.")

    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            # Apply regex compatibility patch for sentence_transformers
            try:
                import regex

                if regex.__version__ == "2.5.159":
                    regex.__version__ = "2025.7.34"
                    if hasattr(regex, "version"):
                        regex.version = "2025.7.34"
            except ImportError:
                pass

            # Import sentence_transformers with warnings suppressed
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*regex.*2019.12.17.*")
                warnings.filterwarnings("ignore", message=".*regex!=2019.12.17.*")

                import sentence_transformers

            print(
                f"Found sentence_transformers version: {sentence_transformers.__version__}"
            )

            # Initialize HuggingFace embeddings with the specified model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print(f"Successfully loaded embedding model: {self.model_name}")
        except ImportError as e:
            # Don't report regex-related errors since they're handled by the patch
            if "regex" not in str(e):
                print(f"Import error (non-critical): {str(e)}")
            self.embeddings = None
        except Exception as e:
            warnings.warn(f"Error initializing embedding model: {e}")
            print(f"Initialization error details: {str(e)}")
            self.embeddings = None

    def _convert_to_langchain_documents(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Convert our custom chunks format to LangChain Document objects.

        Args:
            chunks: List of document chunks in our custom format

        Returns:
            List of LangChain Document objects
        """
        documents = []

        if not chunks:
            print("Warning: No chunks provided to convert to documents")
            return documents

        for chunk in chunks:
            try:
                # Extract text and metadata
                text = chunk.get("text", "")

                # Skip empty text chunks
                if not text:
                    # Try to extract text from nested structure if present
                    if isinstance(chunk.get("chunk"), dict) and chunk["chunk"].get(
                        "text"
                    ):
                        text = chunk["chunk"]["text"]
                    else:
                        continue  # Skip this chunk if no text found

                # Extract all other fields as metadata
                metadata = {k: v for k, v in chunk.items() if k != "text"}

                # Create Document object
                document = Document(page_content=text, metadata=metadata)
                documents.append(document)
            except Exception as e:
                print(f"Error converting chunk to document: {e}")
                continue

        print(f"Converted {len(documents)} valid documents from {len(chunks)} chunks")
        return documents

    def _convert_from_langchain_documents(
        self, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """
        Convert LangChain Document objects back to our custom chunks format.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document chunks in our custom format
        """
        chunks = []

        for doc in documents:
            # Create chunk with text and metadata
            chunk = {"text": doc.page_content}

            # Add metadata
            chunk.update(doc.metadata)

            chunks.append(chunk)

        return chunks

    def build_indexes(self, chunks: List[Dict[str, Any]]):
        """
        Build retrieval indexes from document chunks.

        Args:
            chunks: List of document chunks
        """
        if not langchain_available:
            warnings.warn("LangChain not available. Cannot build indexes.")
            return

        # Store chunks
        self.chunks = chunks

        try:
            # Check if chunks is empty
            if not chunks:
                warnings.warn("No document chunks provided. Cannot build indexes.")
                return

            # Convert chunks to LangChain Document objects
            documents = self._convert_to_langchain_documents(chunks)

            # Check if we have valid documents after conversion
            if not documents:
                warnings.warn(
                    "No valid documents after conversion. Cannot build indexes."
                )
                return

            print(f"Building indexes with {len(documents)} documents")

            # Build dense index if embeddings are available
            if self.embeddings:
                # Create FAISS index
                vector_store = FAISS.from_documents(documents, self.embeddings)
                self.dense_retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 5}
                )
                print("Dense retriever built successfully")

            # Build sparse (BM25) index
            self.sparse_retriever = BM25Retriever.from_documents(documents)
            self.sparse_retriever.k = 5
            print("Sparse retriever built successfully")

            # Create ensemble retriever that combines dense and sparse
            if self.dense_retriever:
                self.retriever = EnsembleRetriever(
                    retrievers=[self.dense_retriever, self.sparse_retriever],
                    weights=[0.5, 0.5],
                )
                print("Hybrid retriever built successfully")
            else:
                # Fall back to sparse retriever
                self.retriever = self.sparse_retriever
                print("Using sparse retriever only")

            self.is_initialized = True
            print("Indexes built successfully")

        except Exception as e:
            warnings.warn(f"Error building indexes: {e}")
            import traceback

            traceback.print_exc()

    def search(
        self, query: str, top_k: int = 5, search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using the specified retrieval method.

        Args:
            query: Search query
            top_k: Number of results to return
            search_type: Retrieval method ("dense", "sparse", or "hybrid")

        Returns:
            List of relevant chunks with scores
        """
        if not self.is_initialized:
            warnings.warn("Indexes not built. Cannot search.")
            return []

        try:
            # Select retriever based on search type
            if search_type == "dense" and self.dense_retriever:
                retriever = self.dense_retriever
                method = "dense"
            elif search_type == "sparse":
                retriever = self.sparse_retriever
                method = "sparse"
            elif search_type == "hybrid" and self.retriever:
                retriever = self.retriever
                method = "hybrid"
            else:
                # Fall back to available retriever
                if self.sparse_retriever:
                    retriever = self.sparse_retriever
                    method = "sparse"
                elif self.dense_retriever:
                    retriever = self.dense_retriever
                    method = "dense"
                else:
                    warnings.warn("No retrievers available.")
                    return []

            # Get relevant documents
            # Handle different retriever types
            if search_type == "hybrid":
                # For EnsembleRetriever, we can't set k directly
                # Configure the individual retrievers if possible
                if hasattr(self.dense_retriever, "search_kwargs"):
                    self.dense_retriever.search_kwargs["k"] = top_k
                if hasattr(self.sparse_retriever, "k"):
                    self.sparse_retriever.k = top_k
            else:
                # For other retrievers, try to set k if the attribute exists
                if hasattr(retriever, "k"):
                    retriever.k = top_k
                elif hasattr(retriever, "search_kwargs"):
                    retriever.search_kwargs["k"] = top_k

            documents = retriever.get_relevant_documents(query)

            # Convert to our format and add scores
            results = []
            for i, doc in enumerate(documents):
                # Calculate a simple score based on position (better retrievers would provide actual scores)
                score = 1.0 - (i * 0.1)
                if score < 0:
                    score = 0.1

                # Create result entry
                chunk = {"text": doc.page_content}
                chunk.update(doc.metadata)

                result = {"chunk": chunk, "score": score, "method": method}

                results.append(result)

            return results

        except Exception as e:
            warnings.warn(f"Error in search: {e}")
            import traceback

            traceback.print_exc()
            return []

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using dense embeddings.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        return self.search(query, top_k, search_type="dense")

    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using sparse embeddings.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        return self.search(query, top_k, search_type="sparse")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        return self.search(query, top_k, search_type="hybrid")

    def save_indexes(self, output_dir: Union[str, Path]):
        """
        Save the indexes to disk.

        Args:
            output_dir: Directory to save the indexes
        """
        if not self.is_initialized:
            warnings.warn("Indexes not built. Nothing to save.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "model_name": self.model_name,
        }

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save chunks
        with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

        # Save FAISS index if available
        if hasattr(self, "dense_retriever") and self.dense_retriever:
            vector_store = self.dense_retriever.vectorstore
            if hasattr(vector_store, "save_local"):
                vector_store_dir = output_dir / "vector_store"
                vector_store_dir.mkdir(exist_ok=True)
                vector_store.save_local(str(vector_store_dir))
                print(f"Vector store saved to {vector_store_dir}")

        print(f"Indexes saved to {output_dir}")

    def load_indexes(self, input_dir: Union[str, Path]):
        """
        Load the indexes from disk.

        Args:
            input_dir: Directory containing the saved indexes
        """
        if not langchain_available:
            warnings.warn("LangChain not available. Cannot load indexes.")
            return

        input_dir = Path(input_dir)

        try:
            # Load configuration
            with open(input_dir / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)

            # Update model name
            self.model_name = config.get("model_name", self.model_name)

            # Initialize embeddings
            self._initialize_embeddings()

            # Load chunks
            with open(input_dir / "chunks.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            # Load FAISS index if available
            vector_store_dir = input_dir / "vector_store"
            if vector_store_dir.exists() and self.embeddings:
                try:
                    vector_store = FAISS.load_local(
                        str(vector_store_dir), self.embeddings
                    )
                    self.dense_retriever = vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 5}
                    )
                    print("Dense retriever loaded successfully")
                except Exception as e:
                    warnings.warn(f"Error loading vector store: {e}")

            # Create documents from chunks for BM25
            documents = self._convert_to_langchain_documents(self.chunks)

            # Create BM25 retriever
            self.sparse_retriever = BM25Retriever.from_documents(documents)
            self.sparse_retriever.k = 5
            print("Sparse retriever loaded successfully")

            # Create ensemble retriever if both are available
            if self.dense_retriever:
                self.retriever = EnsembleRetriever(
                    retrievers=[self.dense_retriever, self.sparse_retriever],
                    weights=[0.5, 0.5],
                )
                print("Hybrid retriever loaded successfully")
            else:
                # Fall back to sparse retriever
                self.retriever = self.sparse_retriever
                print("Using sparse retriever only")

            self.is_initialized = True
            print(f"Indexes loaded from {input_dir}")

        except Exception as e:
            warnings.warn(f"Error loading indexes: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    embedding_manager = LangChainEmbeddingManager(model_name="all-MiniLM-L6-v2")

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
    embedding_manager.build_indexes(chunks)

    # Search example
    results = embedding_manager.hybrid_search(
        "What was the revenue in Q2 2023?", top_k=2
    )
    for result in results:
        print(f"Score: {result['score']}, Method: {result['method']}")
        print(f"Text: {result['chunk']['text']}")
