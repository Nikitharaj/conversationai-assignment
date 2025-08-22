import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class DocumentProcessor:
    """Class for processing financial documents into clean text."""

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the document processor.

        Args:
            output_dir: Directory to save processed documents
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_document(self, file_path: Union[str, Path]) -> str:
        """
        Process a document based on its file extension.

        Args:
            file_path: Path to the document file

        Returns:
            Processed text content
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process based on file extension
        if file_path.suffix.lower() == ".pdf":
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return self._process_excel(file_path)
        elif file_path.suffix.lower() == ".html":
            return self._process_html(file_path)
        elif file_path.suffix.lower() == ".csv":
            return self._process_csv(file_path)
        elif file_path.suffix.lower() == ".txt":
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _process_pdf(self, file_path: Path) -> str:
        """
        Process PDF files using PyPDF2 and OCR if needed.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted and cleaned text
        """
        # Try to extract text directly first
        text = self._extract_text_from_pdf(file_path)

        # If text extraction fails or returns very little text, try OCR
        if not text or len(text.strip()) < 100:
            text = self._ocr_pdf(file_path)

        # Clean the extracted text
        cleaned_text = self._clean_text(text)

        # Save processed text
        output_file = self.output_dir / f"{file_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        return cleaned_text

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF2."""
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text

    def _ocr_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using OCR."""
        text = ""
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image) + "\n\n"
        return text

    def _process_excel(self, file_path: Path) -> str:
        """
        Process Excel files.

        Args:
            file_path: Path to the Excel file

        Returns:
            Extracted and formatted text
        """
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        all_text = []

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Add sheet name as a section header
            all_text.append(f"## {sheet_name}")

            # Convert dataframe to string representation
            table_str = df.to_string(index=False)
            all_text.append(table_str)
            all_text.append("\n")

        combined_text = "\n".join(all_text)

        # Save processed text
        output_file = self.output_dir / f"{file_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        return combined_text

    def _process_html(self, file_path: Path) -> str:
        """
        Process HTML files.

        Args:
            file_path: Path to the HTML file

        Returns:
            Extracted and cleaned text
        """
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text()

        # Clean text
        cleaned_text = self._clean_text(text)

        # Save processed text
        output_file = self.output_dir / f"{file_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        return cleaned_text

    def _process_csv(self, file_path: Path) -> str:
        """
        Process CSV files.

        Args:
            file_path: Path to the CSV file

        Returns:
            Formatted text representation
        """
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)

        # Save processed text
        output_file = self.output_dir / f"{file_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        return text

    def _process_text(self, file_path: Path) -> str:
        """
        Process plain text files.

        Args:
            file_path: Path to the text file

        Returns:
            Cleaned text content
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Clean the text
        cleaned_text = self._clean_text(text)

        # Save processed text (might be the same as input if already clean)
        output_file = self.output_dir / f"{file_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        return cleaned_text

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing headers, footers, page numbers, etc.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Remove headers/footers (common patterns)
        text = re.sub(r"^\s*Page \d+ of \d+.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*Confidential.*$", "", text, flags=re.MULTILINE)

        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)

        return text.strip()

    def segment_by_section(self, text: str) -> Dict[str, str]:
        """
        Segment the document into sections based on headings.

        Args:
            text: Processed document text

        Returns:
            Dictionary mapping section names to their content
        """
        # Common financial statement section headers
        section_patterns = [
            r"(?i)income statement",
            r"(?i)statement of (comprehensive )?income",
            r"(?i)balance sheet",
            r"(?i)statement of financial position",
            r"(?i)cash flow",
            r"(?i)statement of cash flows",
            r"(?i)statement of changes in equity",
            r"(?i)notes to( the)? financial statements",
            r"(?i)management discussion and analysis",
            r"(?i)risk factors",
        ]

        # Find all potential section headers
        sections = {}
        current_section = "General"
        current_content = []

        for line in text.split("\n"):
            # Check if the line matches any section pattern
            is_section_header = False
            for pattern in section_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Save the previous section
                    if current_content:
                        sections[current_section] = "\n".join(current_content)

                    # Start a new section
                    current_section = line.strip()
                    current_content = []
                    is_section_header = True
                    break

            if not is_section_header:
                current_content.append(line)

        # Save the last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def batch_process(self, directory: Union[str, Path]) -> List[str]:
        """
        Process all supported files in a directory.

        Args:
            directory: Directory containing documents to process

        Returns:
            List of paths to processed text files
        """
        directory = Path(directory)
        supported_extensions = [".pdf", ".xlsx", ".xls", ".html", ".csv", ".txt"]
        processed_files = []

        for file_path in directory.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    self.process_document(file_path)
                    processed_files.append(
                        str(self.output_dir / f"{file_path.stem}.txt")
                    )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return processed_files


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(output_dir="../../data/processed")

    # Process a single file
    # processor.process_document("../../data/raw/example.pdf")

    # Process all files in a directory
    # processor.batch_process("../../data/raw")
