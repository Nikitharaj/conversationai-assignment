import os
import json
from pathlib import Path
from typing import List, Dict, Union, Optional
import re


class DocumentChunker:
    """Class for chunking documents into smaller segments for retrieval."""

    def __init__(self, chunk_sizes: List[int] = [100, 400], chunk_overlap: int = 50):
        """
        Initialize the document chunker.

        Args:
            chunk_sizes: List of token sizes for chunking (e.g., [100, 400])
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, text: str, chunk_size: int
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Split a document into chunks of approximately the specified token size.

        Args:
            text: Document text to chunk
            chunk_size: Target size of each chunk in tokens

        Returns:
            List of dictionaries containing chunks and metadata
        """
        # Simple tokenization by splitting on whitespace
        tokens = text.split()

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))

            # Extract the chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Convert back to text
            chunk_text = " ".join(chunk_tokens)

            # Store the chunk with metadata
            chunks.append(
                {
                    "text": chunk_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx - 1,
                    "chunk_size": end_idx - start_idx,
                }
            )

            # Move to the next chunk, considering overlap
            start_idx += chunk_size - self.chunk_overlap

        return chunks

    def chunk_document_by_section(
        self, sections: Dict[str, str], chunk_size: int
    ) -> List[Dict[str, Union[str, int, str]]]:
        """
        Split a document into chunks by section.

        Args:
            sections: Dictionary mapping section names to their content
            chunk_size: Target size of each chunk in tokens

        Returns:
            List of dictionaries containing chunks and metadata
        """
        all_chunks = []

        for section_name, section_text in sections.items():
            # Chunk the section
            section_chunks = self.chunk_document(section_text, chunk_size)

            # Add section metadata to each chunk
            for chunk in section_chunks:
                chunk["section"] = section_name
                all_chunks.append(chunk)

        return all_chunks

    def chunk_all_sizes(self, text: str) -> Dict[int, List[Dict[str, Union[str, int]]]]:
        """
        Chunk a document into all specified chunk sizes.

        Args:
            text: Document text to chunk

        Returns:
            Dictionary mapping chunk sizes to their respective chunks
        """
        result = {}

        for size in self.chunk_sizes:
            result[size] = self.chunk_document(text, size)

        return result

    def chunk_file(
        self, file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[int, List[Dict[str, Union[str, int]]]]:
        """
        Chunk a text file and optionally save the chunks.

        Args:
            file_path: Path to the text file
            output_dir: Optional directory to save the chunks

        Returns:
            Dictionary mapping chunk sizes to their respective chunks
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the text file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk the document
        chunks_by_size = self.chunk_all_sizes(text)

        # Save the chunks if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for size, chunks in chunks_by_size.items():
                output_file = output_dir / f"{file_path.stem}_chunks_{size}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, indent=2)

        return chunks_by_size

    def batch_chunk_files(
        self, directory: Union[str, Path], output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Dict[int, List[Dict[str, Union[str, int]]]]]:
        """
        Chunk all text files in a directory.

        Args:
            directory: Directory containing text files
            output_dir: Optional directory to save the chunks

        Returns:
            Dictionary mapping file names to their chunks by size
        """
        directory = Path(directory)
        results = {}

        for file_path in directory.glob("*.txt"):
            try:
                results[file_path.stem] = self.chunk_file(file_path, output_dir)
            except Exception as e:
                print(f"Error chunking {file_path}: {e}")

        return results


if __name__ == "__main__":
    # Example usage
    chunker = DocumentChunker(chunk_sizes=[100, 400], chunk_overlap=50)

    # Chunk a single file
    # chunker.chunk_file("../../data/processed/example.txt", output_dir="../../data/chunks")

    # Chunk all files in a directory
    # chunker.batch_chunk_files("../../data/processed", output_dir="../../data/chunks")
