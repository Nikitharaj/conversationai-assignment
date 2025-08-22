"""
Tests for the DocumentProcessor class.
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

from src.data_processing.document_processor import DocumentProcessor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the DocumentProcessor class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "processed"
        self.output_dir.mkdir(exist_ok=True)
        self.processor = DocumentProcessor(output_dir=self.output_dir)

        # Create test files
        self.test_txt_path = Path(self.temp_dir) / "test.txt"
        with open(self.test_txt_path, "w") as f:
            f.write(
                "This is a test document.\nIt has multiple lines.\nIt is used for testing."
            )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test initialization of DocumentProcessor."""
        self.assertEqual(self.processor.output_dir, self.output_dir)
        self.assertTrue(self.output_dir.exists())

    def test_process_text_file(self):
        """Test processing a text file."""
        result = self.processor.process_document(self.test_txt_path)
        self.assertIsInstance(result, str)
        self.assertIn("This is a test document", result)
        self.assertIn("It has multiple lines", result)

    def test_process_pdf_file(self):
        """Test processing a PDF file."""
        # Create a simple test that doesn't rely on external dependencies
        test_pdf_path = Path(self.temp_dir) / "test.pdf"
        with open(test_pdf_path, "wb") as f:
            f.write(b"%PDF-1.5\nsome dummy pdf content")

        # Create a complete mock of the _process_pdf method
        with patch.object(
            self.processor,
            "_process_pdf",
            return_value="Page 1 content\n\nPage 2 content",
        ) as mock_process:
            result = self.processor.process_document(test_pdf_path)

            # Verify the method was called
            mock_process.assert_called_once()

            # Check the result
            self.assertIsInstance(result, str)
            self.assertEqual(result, "Page 1 content\n\nPage 2 content")

    @patch("pandas.ExcelFile")
    @patch("pandas.read_excel")
    def test_process_excel_file(self, mock_read_excel, mock_excel_file):
        """Test processing an Excel file."""
        # Mock ExcelFile
        mock_excel_instance = MagicMock()
        mock_excel_file.return_value = mock_excel_instance
        mock_excel_instance.sheet_names = ["Sheet1", "Sheet2"]

        # Mock pandas read_excel
        mock_df = MagicMock()
        mock_read_excel.return_value = mock_df
        mock_df.to_string.return_value = "Excel content as string"

        test_excel_path = Path(self.temp_dir) / "test.xlsx"
        with open(test_excel_path, "wb") as f:
            f.write(b"dummy excel content")

        result = self.processor.process_document(test_excel_path)
        self.assertIsInstance(result, str)
        self.assertIn("Excel content as string", result)

    @patch("pandas.read_csv")
    def test_process_csv_file(self, mock_read_csv):
        """Test processing a CSV file."""
        # Mock pandas read_csv
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df
        mock_df.to_string.return_value = "CSV content as string"

        test_csv_path = Path(self.temp_dir) / "test.csv"
        with open(test_csv_path, "w") as f:
            f.write("dummy csv content")

        result = self.processor.process_document(test_csv_path)
        self.assertIsInstance(result, str)
        self.assertIn("CSV content as string", result)

    @patch("bs4.BeautifulSoup")
    def test_process_html_file(self, mock_bs):
        """Test processing an HTML file."""
        # Mock BeautifulSoup
        mock_instance = MagicMock()
        mock_bs.return_value = mock_instance
        mock_instance.get_text.return_value = "dummy html content"

        test_html_path = Path(self.temp_dir) / "test.html"
        with open(test_html_path, "w") as f:
            f.write("<html><body>dummy html content</body></html>")

        result = self.processor.process_document(test_html_path)
        self.assertIsInstance(result, str)
        self.assertIn("dummy html content", result)

    def test_unsupported_file_type(self):
        """Test processing an unsupported file type."""
        test_unsupported_path = Path(self.temp_dir) / "test.xyz"
        with open(test_unsupported_path, "w") as f:
            f.write("dummy unsupported content")

        with self.assertRaises(ValueError):
            self.processor.process_document(test_unsupported_path)

    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "This is a test\n\n\nwith   multiple    spaces\t\tand tabs."
        clean_text = self.processor._clean_text(dirty_text)
        # Check that whitespace is normalized but not completely removed
        self.assertIn("This is a test", clean_text)
        self.assertIn("with multiple spaces", clean_text)
        self.assertIn("and tabs", clean_text)


if __name__ == "__main__":
    unittest.main()
