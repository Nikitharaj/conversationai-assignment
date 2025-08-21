import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
import pickle

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class EmbeddingManager:
    """Class for managing document embeddings and retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dense_index = None
        self.sparse_index = None
        self.chunks = []
        self.chunk_texts = []
        self.tfidf_vectorizer = None
        self.bm25 = None

    def create_dense_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create dense embeddings for document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            Numpy array of embeddings
        """
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]

        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)

        return embeddings

    def create_sparse_index(
        self, chunks: List[Dict[str, Any]], method: str = "bm25"
    ) -> Any:
        """
        Create a sparse index for document chunks.

        Args:
            chunks: List of document chunks
            method: Sparse indexing method ("bm25" or "tfidf")

        Returns:
            Sparse index (BM25 or TF-IDF)
        """
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]

        if method == "bm25":
            # Tokenize texts for BM25
            tokenized_texts = [word_tokenize(text.lower()) for text in texts]

            # Create BM25 index
            bm25 = BM25Okapi(tokenized_texts)
            return bm25

        elif method == "tfidf":
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer()

            # Fit and transform texts
            tfidf_matrix = vectorizer.fit_transform(texts)

            return {"vectorizer": vectorizer, "matrix": tfidf_matrix}

        else:
            raise ValueError(f"Unsupported sparse indexing method: {method}")

    def build_indexes(self, chunks: List[Dict[str, Any]]):
        """
        Build both dense and sparse indexes for document chunks.

        Args:
            chunks: List of document chunks
        """
        # Store chunks and extract texts
        self.chunks = chunks
        self.chunk_texts = [chunk["text"] for chunk in chunks]

        # Create dense embeddings
        embeddings = self.create_dense_embeddings(chunks)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(vector_dimension)
        self.dense_index.add(embeddings)

        # Create BM25 sparse index
        self.bm25 = self.create_sparse_index(chunks, method="bm25")

        # Create TF-IDF sparse index
        tfidf_result = self.create_sparse_index(chunks, method="tfidf")
        self.tfidf_vectorizer = tfidf_result["vectorizer"]
        self.tfidf_matrix = tfidf_result["matrix"]

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using dense embeddings.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        if self.dense_index is None:
            raise ValueError("Dense index not built. Call build_indexes first.")

        # Encode the query
        query_embedding = self.model.encode([query])

        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search the index
        scores, indices = self.dense_index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:  # Valid index check
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "score": float(scores[0][i]),
                        "method": "dense",
                    }
                )

        return results

    def sparse_search(
        self, query: str, top_k: int = 5, method: str = "bm25"
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using sparse indexing.

        Args:
            query: Search query
            top_k: Number of results to return
            method: Sparse indexing method ("bm25" or "tfidf")

        Returns:
            List of relevant chunks with scores
        """
        if method == "bm25":
            if self.bm25 is None:
                raise ValueError("BM25 index not built. Call build_indexes first.")

            # Tokenize query
            tokenized_query = word_tokenize(query.lower())

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Prepare results
            results = []
            for idx in top_indices:
                if idx < len(self.chunks) and idx >= 0:  # Valid index check
                    results.append(
                        {
                            "chunk": self.chunks[idx],
                            "score": float(scores[idx]),
                            "method": "bm25",
                        }
                    )

            return results

        elif method == "tfidf":
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF index not built. Call build_indexes first.")

            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            # Prepare results
            results = []
            for idx in top_indices:
                if idx < len(self.chunks) and idx >= 0:  # Valid index check
                    results.append(
                        {
                            "chunk": self.chunks[idx],
                            "score": float(similarities[idx]),
                            "method": "tfidf",
                        }
                    )

            return results

        else:
            raise ValueError(f"Unsupported sparse indexing method: {method}")

    def hybrid_search(
        self, query: str, top_k: int = 5, method: str = "fusion"
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            method: Hybrid method ("fusion" or "union")

        Returns:
            List of relevant chunks with scores
        """
        # Get dense search results
        dense_results = self.dense_search(query, top_k=top_k)

        # Get sparse search results (BM25)
        sparse_results = self.sparse_search(query, top_k=top_k, method="bm25")

        if method == "fusion":
            # Score fusion (reciprocal rank fusion)
            all_results = {}

            # Process dense results
            for rank, result in enumerate(dense_results):
                # Use a more reliable way to identify chunks
                chunk_id = id(result["chunk"])
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        "chunk": result["chunk"],
                        "score": 0,
                        "method": "hybrid",
                    }
                all_results[chunk_id]["score"] += 1.0 / (rank + 1)

            # Process sparse results
            for rank, result in enumerate(sparse_results):
                # Use a more reliable way to identify chunks
                chunk_id = id(result["chunk"])
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        "chunk": result["chunk"],
                        "score": 0,
                        "method": "hybrid",
                    }
                all_results[chunk_id]["score"] += 1.0 / (rank + 1)

            # Sort by score and return top k
            results = list(all_results.values())
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        elif method == "union":
            # Union of results
            seen_chunks = set()
            results = []

            # Add dense results first
            for result in dense_results:
                # Use a more reliable way to identify chunks
                chunk_id = id(result["chunk"])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    results.append(result)

            # Add sparse results if not already included
            for result in sparse_results:
                # Use a more reliable way to identify chunks
                chunk_id = id(result["chunk"])
                if chunk_id not in seen_chunks and len(results) < top_k:
                    seen_chunks.add(chunk_id)
                    results.append(result)

            return results[:top_k]

        else:
            raise ValueError(f"Unsupported hybrid method: {method}")

    def save_indexes(self, output_dir: Union[str, Path]):
        """
        Save the indexes to disk.

        Args:
            output_dir: Directory to save the indexes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks
        with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

        # Save dense index
        if self.dense_index is not None:
            faiss.write_index(self.dense_index, str(output_dir / "dense_index.faiss"))

        # Save BM25 index
        if self.bm25 is not None:
            with open(output_dir / "bm25_index.pkl", "wb") as f:
                pickle.dump(self.bm25, f)

        # Save TF-IDF vectorizer and matrix
        if self.tfidf_vectorizer is not None:
            with open(output_dir / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

            with open(output_dir / "tfidf_matrix.pkl", "wb") as f:
                pickle.dump(self.tfidf_matrix, f)

    def load_indexes(self, input_dir: Union[str, Path]):
        """
        Load the indexes from disk.

        Args:
            input_dir: Directory containing the saved indexes
        """
        input_dir = Path(input_dir)

        # Load chunks
        with open(input_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            self.chunk_texts = [chunk["text"] for chunk in self.chunks]

        # Load dense index
        if (input_dir / "dense_index.faiss").exists():
            self.dense_index = faiss.read_index(str(input_dir / "dense_index.faiss"))

        # Load BM25 index
        if (input_dir / "bm25_index.pkl").exists():
            with open(input_dir / "bm25_index.pkl", "rb") as f:
                self.bm25 = pickle.load(f)

        # Load TF-IDF vectorizer and matrix
        if (input_dir / "tfidf_vectorizer.pkl").exists():
            with open(input_dir / "tfidf_vectorizer.pkl", "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

            with open(input_dir / "tfidf_matrix.pkl", "rb") as f:
                self.tfidf_matrix = pickle.load(f)


if __name__ == "__main__":
    # Example usage
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # Build indexes from chunks
    # with open("../../data/chunks/example_chunks_400.json", 'r', encoding='utf-8') as f:
    #     chunks = json.load(f)
    #     embedding_manager.build_indexes(chunks)

    # Save indexes
    # embedding_manager.save_indexes("../../data/indexes")

    # Load indexes
    # embedding_manager.load_indexes("../../data/indexes")

    # Search example
    # results = embedding_manager.hybrid_search("What was the revenue in Q2 2023?", top_k=3)
    # for result in results:
    #     print(f"Score: {result['score']}, Method: {result['method']}")
    #     print(result['chunk']['text'][:100] + "...")
