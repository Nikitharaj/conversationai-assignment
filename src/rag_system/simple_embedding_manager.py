"""
Simplified version of the embedding manager that doesn't rely on external libraries.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
import pickle
import random


class SimpleEmbeddingManager:
    """Simplified class for managing document embeddings and retrieval."""

    def __init__(self, model_name: str = "mock-model"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the model (not used in this simplified version)
        """
        self.model_name = model_name
        self.chunks = []
        self.chunk_texts = []

    def build_indexes(self, chunks: List[Dict[str, Any]]):
        """
        Build indexes for document chunks.

        Args:
            chunks: List of document chunks
        """
        # Store chunks and extract texts
        self.chunks = chunks
        self.chunk_texts = [chunk["text"] for chunk in chunks]

        print(f"Built indexes for {len(chunks)} chunks")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks:
            raise ValueError("No chunks indexed. Call build_indexes first.")

        # This is a mock implementation that returns random chunks with random scores
        # In a real implementation, this would use actual embeddings and similarity search

        # Get a random sample of chunks (or all if there are fewer than top_k)
        num_results = min(top_k, len(self.chunks))
        selected_indices = random.sample(range(len(self.chunks)), num_results)

        results = []
        for i, idx in enumerate(selected_indices):
            # Generate a random score between 0.5 and 1.0
            score = 0.5 + random.random() * 0.5

            results.append(
                {"chunk": self.chunks[idx], "score": score, "method": "hybrid"}
            )

        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simplified dense search that just calls hybrid_search."""
        results = self.hybrid_search(query, top_k)
        for result in results:
            result["method"] = "dense"
        return results

    def sparse_search(
        self, query: str, top_k: int = 5, method: str = "bm25"
    ) -> List[Dict[str, Any]]:
        """Simplified sparse search that just calls hybrid_search."""
        results = self.hybrid_search(query, top_k)
        for result in results:
            result["method"] = method
        return results

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

        print(f"Saved indexes to {output_dir}")

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

        print(f"Loaded indexes from {input_dir}")
