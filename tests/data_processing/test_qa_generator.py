"""
Tests for the QAGenerator class.
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
import json
from pathlib import Path
import tempfile
import shutil

import pytest
from unittest.mock import patch, MagicMock

from src.data_processing.qa_generator import QAGenerator


class TestQAGenerator(unittest.TestCase):
    """Test cases for the QAGenerator class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "qa_pairs"
        self.output_dir.mkdir(exist_ok=True)
        self.generator = QAGenerator(output_dir=self.output_dir)

        # Create test files
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.processed_dir.mkdir(exist_ok=True)

        self.test_doc_path = self.processed_dir / "test_doc.txt"
        with open(self.test_doc_path, "w") as f:
            f.write(
                "This is a financial document. The revenue for Q2 2023 was $10 million. "
                "The profit margin was 15%. The company has 500 employees. "
                "The debt to equity ratio is 0.8. The return on investment was 12%."
            )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test initialization of QAGenerator."""
        self.assertEqual(self.generator.output_dir, self.output_dir)
        self.assertTrue(self.output_dir.exists())

    def test_extract_financial_entities(self):
        """Test extraction of financial entities."""
        text = "The revenue was $10 million and profit was 15%."
        entities = self.generator._extract_financial_entities(text)
        self.assertIsInstance(entities, dict)
        self.assertIn("amounts", entities)
        self.assertIn("percentages", entities)
        self.assertIn("$10 million", entities["amounts"])
        self.assertIn("15%", entities["percentages"])

    def test_generate_qa_pairs_from_text(self):
        """Test generation of Q&A pairs from text."""
        text = "The revenue for Q2 2023 was $10 million. The profit margin was 15%."
        qa_pairs = self.generator.generate_qa_pairs_from_text(text)
        self.assertIsInstance(qa_pairs, list)
        self.assertTrue(len(qa_pairs) > 0)

        # Check structure of Q&A pairs
        for qa_pair in qa_pairs:
            self.assertIn("question", qa_pair)
            self.assertIn("answer", qa_pair)
            self.assertIsInstance(qa_pair["question"], str)
            self.assertIsInstance(qa_pair["answer"], str)

    def test_generate_qa_pairs_from_file(self):
        """Test generation of Q&A pairs from a file."""
        qa_pairs = self.generator.generate_qa_pairs_from_file(self.test_doc_path)
        self.assertIsInstance(qa_pairs, list)
        self.assertTrue(len(qa_pairs) > 0)

        # Check structure of Q&A pairs
        for qa_pair in qa_pairs:
            self.assertIn("question", qa_pair)
            self.assertIn("answer", qa_pair)
            self.assertIsInstance(qa_pair["question"], str)
            self.assertIsInstance(qa_pair["answer"], str)

    def test_generate_qa_pairs_from_directory(self):
        """Test generation of Q&A pairs from a directory."""
        # Create additional test files
        for i in range(2):
            with open(self.processed_dir / f"test_doc_{i}.txt", "w") as f:
                f.write(f"This is test document {i}. It contains financial data.")

        qa_pairs = self.generator.generate_qa_pairs_from_directory(self.processed_dir)
        self.assertIsInstance(qa_pairs, list)
        self.assertTrue(len(qa_pairs) > 0)

        # Check structure of Q&A pairs
        for qa_pair in qa_pairs:
            self.assertIn("question", qa_pair)
            self.assertIn("answer", qa_pair)
            self.assertIsInstance(qa_pair["question"], str)
            self.assertIsInstance(qa_pair["answer"], str)

    def test_save_qa_pairs(self):
        """Test saving Q&A pairs to a file."""
        qa_pairs = [
            {"question": "What was the revenue?", "answer": "$10 million"},
            {"question": "What was the profit margin?", "answer": "15%"},
        ]

        output_file = self.generator.save_qa_pairs(qa_pairs, "test_qa_pairs")
        self.assertTrue(output_file.exists())

        # Check if the saved file contains the correct data
        with open(output_file, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(len(saved_data), len(qa_pairs))
        self.assertEqual(saved_data[0]["question"], qa_pairs[0]["question"])
        self.assertEqual(saved_data[0]["answer"], qa_pairs[0]["answer"])

    def test_generate_question_templates(self):
        """Test generation of question templates."""
        templates = self.generator._generate_question_templates()
        self.assertIsInstance(templates, dict)
        self.assertIn("amount", templates)
        self.assertIn("percentage", templates)
        self.assertTrue(len(templates["amount"]) > 0)
        self.assertTrue(len(templates["percentage"]) > 0)

    def test_generate_irrelevant_questions(self):
        """Test generation of irrelevant questions."""
        irrelevant_questions = self.generator._generate_irrelevant_questions(5)
        self.assertIsInstance(irrelevant_questions, list)
        self.assertEqual(len(irrelevant_questions), 5)
        for question in irrelevant_questions:
            self.assertIsInstance(question, str)


if __name__ == "__main__":
    unittest.main()
