"""
LangChain-based document chunker for splitting documents into retrievable segments.

This module replaces the custom DocumentChunker with LangChain's text splitters.
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple

# Try to import LangChain components
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter,
    )

    langchain_available = True
except ImportError:
    # Only warn if this is not being imported during test execution
    import sys

    if not any("pytest" in arg for arg in sys.argv) and not any(
        "unittest" in arg for arg in sys.argv
    ):
        warnings.warn(
            "LangChain text splitters not available. Install with 'pip install langchain-text-splitters'"
        )
    langchain_available = False


class DocumentChunker:
    """LangChain-based document chunker for splitting documents into retrievable segments."""

    def __init__(
        self,
        chunk_sizes: List[int] = [100, 400],
        chunk_overlap: int = 50,
        splitter_type: str = "recursive",
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_sizes: List of token/character sizes for chunking (e.g., [100, 400])
            chunk_overlap: Number of tokens/characters to overlap between chunks
            splitter_type: Type of splitter to use ("recursive", "character", or "token")
        """
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type
        self.default_chunk_size = (
            chunk_sizes[1] if len(chunk_sizes) > 1 else chunk_sizes[0]
        )

        # Initialize text splitters if LangChain is available
        if langchain_available:
            self._initialize_splitters()
        else:
            self.splitters = {}
            # Only warn if this is not being imported during test execution
            import sys

            if not any("pytest" in arg for arg in sys.argv) and not any(
                "unittest" in arg for arg in sys.argv
            ):
                warnings.warn(
                    "Using fallback chunking method since LangChain is not available."
                )

    def _initialize_splitters(self):
        """Initialize text splitters for each chunk size."""
        self.splitters = {}

        for size in self.chunk_sizes:
            if self.splitter_type == "recursive":
                # RecursiveCharacterTextSplitter is the most versatile
                self.splitters[size] = RecursiveCharacterTextSplitter(
                    chunk_size=size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            elif self.splitter_type == "character":
                # CharacterTextSplitter is simpler but effective
                self.splitters[size] = CharacterTextSplitter(
                    chunk_size=size, chunk_overlap=self.chunk_overlap, separator="\n"
                )
            elif self.splitter_type == "token":
                # TokenTextSplitter uses tokens instead of characters
                self.splitters[size] = TokenTextSplitter(
                    chunk_size=size, chunk_overlap=self.chunk_overlap
                )
            else:
                # Default to recursive
                self.splitters[size] = RecursiveCharacterTextSplitter(
                    chunk_size=size, chunk_overlap=self.chunk_overlap
                )

    def _fallback_chunk_document(
        self,
        text: str,
        chunk_size: int,
        source_file: str = "unknown",
        section: str = "unknown",
        year: Optional[int] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Fallback method for chunking when LangChain is not available.

        Args:
            text: Document text to chunk
            chunk_size: Target size of each chunk in tokens/characters

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

            # Generate unique chunk ID
            chunk_id = f"{source_file}_{section}_{year or 'unknown'}_{chunk_size}_{len(chunks)}"

            # Store the chunk with comprehensive metadata (Group 118)
            chunks.append(
                {
                    "id": chunk_id,  # Unique chunk identifier
                    "text": chunk_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx - 1,
                    "token_count": end_idx - start_idx,  # Token count
                    "chunk_size": end_idx - start_idx,  # Token count (keeping original)
                    "char_count": len(chunk_text),  # Character count
                    "chunk_index": len(chunks),
                    # Group 118 required metadata
                    "section": section,
                    "year": year,
                    "source_file": source_file,
                    "chunk_method": "fallback_token_based",
                    "target_chunk_size": chunk_size,
                }
            )

            # Move to the next chunk, considering overlap
            start_idx += chunk_size - self.chunk_overlap

        return chunks

    def chunk_document(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        source_file: str = "unknown",
        section: str = "unknown",
        year: Optional[int] = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Split a document into chunks with comprehensive metadata (Group 118 requirements).

        Args:
            text: Document text to chunk
            chunk_size: Target size of each chunk (uses default if not specified)
            source_file: Name of the source file
            section: Financial section (Income Statement, Balance Sheet, Cash Flow, Notes, MD&A)
            year: Fiscal year of the document

        Returns:
            List of dictionaries containing chunks and comprehensive metadata
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size

        # Use LangChain if available
        if langchain_available and self.splitters:
            try:
                # Get the appropriate splitter
                splitter = self.splitters.get(
                    chunk_size,
                    RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=self.chunk_overlap
                    ),
                )

                # Split the text into documents
                langchain_docs = splitter.create_documents([text])

                # Convert to our format
                chunks = []
                for i, doc in enumerate(langchain_docs):
                    # Calculate approximate token positions
                    start_idx = i * (chunk_size - self.chunk_overlap) if i > 0 else 0
                    end_idx = start_idx + len(doc.page_content.split())

                    # Generate unique chunk ID
                    chunk_id = (
                        f"{source_file}_{section}_{year or 'unknown'}_{chunk_size}_{i}"
                    )

                    chunks.append(
                        {
                            "id": chunk_id,  # Unique chunk identifier
                            "text": doc.page_content,
                            "start_idx": start_idx,
                            "end_idx": end_idx - 1,
                            "token_count": len(
                                doc.page_content.split()
                            ),  # Approximate token count
                            "chunk_size": len(
                                doc.page_content.split()
                            ),  # Token count (keeping original)
                            "char_count": len(doc.page_content),  # Character count
                            "chunk_index": i,
                            "total_chunks": len(langchain_docs),
                            # Group 118 required metadata
                            "section": section,
                            "year": year,
                            "source_file": source_file,
                            "chunk_method": f"langchain_{self.splitter_type}",
                            "target_chunk_size": chunk_size,
                        }
                    )

                return chunks
            except Exception as e:
                warnings.warn(
                    f"Error using LangChain text splitter: {e}. Falling back to basic chunking."
                )
                return self._fallback_chunk_document(
                    text, chunk_size, source_file, section, year
                )
        else:
            # Use fallback method
            return self._fallback_chunk_document(
                text, chunk_size, source_file, section, year
            )

    def chunk_document_by_section(
        self, sections: Dict[str, str], chunk_size: Optional[int] = None
    ) -> List[Dict[str, Union[str, int, str]]]:
        """
        Split a document into chunks by section.

        Args:
            sections: Dictionary mapping section names to their content
            chunk_size: Target size of each chunk (uses default if not specified)

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

    # Example text
    text = """
    LangChain is a framework for developing applications powered by language models. 
    It enables applications that are:
    
    1. Data-aware: connect a language model to other sources of data
    2. Agentic: allow a language model to interact with its environment
    
    The main value props of LangChain are:
    
    1. Components: abstractions for working with language models, along with a collection of implementations for each abstraction. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not
    2. Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level tasks
    
    Off-the-shelf chains make it easy to get started. Components make it easy to customize existing chains and build new ones.
    """

    # Chunk the document
    chunks = chunker.chunk_document(text)

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk['text'][:50]}...")
