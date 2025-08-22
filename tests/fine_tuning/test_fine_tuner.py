"""
Tests for the FineTuner class.
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

import os
import unittest
from pathlib import Path
import tempfile
import shutil
import json
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", message=".*regex.*2019.12.17.*")
warnings.filterwarnings("ignore", message=".*regex!=2019.12.17.*")
warnings.filterwarnings(
    "ignore", message=".*bitsandbytes.*compiled without GPU support.*"
)
warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")

import pytest
from unittest.mock import patch, MagicMock

from src.fine_tuning.fine_tuner import FineTuner


class TestFineTuner(unittest.TestCase):
    """Test cases for the FineTuner class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "model"
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Create test QA pairs
        self.qa_pairs = [
            {"question": "What was the revenue?", "answer": "$10 million"},
            {"question": "What was the profit margin?", "answer": "15%"},
        ]

        # Save test QA pairs
        with open(self.data_dir / "qa_pairs.json", "w") as f:
            json.dump(self.qa_pairs, f)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @patch("src.fine_tuning.fine_tuner.HuggingFacePipeline")
    def test_initialization(self, mock_hf_pipeline):
        """Test initialization of FineTuner."""
        # Mock HuggingFacePipeline
        mock_instance = MagicMock()
        mock_hf_pipeline.return_value = mock_instance

        # Force langchain_available and transformers_available to be True for this test
        with (
            patch("src.fine_tuning.fine_tuner.langchain_available", True),
            patch("src.fine_tuning.fine_tuner.transformers_available", True),
        ):
            tuner = FineTuner(model_name="distilgpt2")
            self.assertEqual(tuner.model_name, "distilgpt2")
            self.assertIsNone(tuner.model)
            self.assertIsNone(tuner.tokenizer)
            self.assertFalse(tuner.model_loaded)

    @patch("src.fine_tuning.fine_tuner.HuggingFacePipeline")
    @patch("src.fine_tuning.fine_tuner.AutoModelForCausalLM")
    @patch("src.fine_tuning.fine_tuner.AutoTokenizer")
    def test_load_model(self, mock_tokenizer, mock_model, mock_hf_pipeline):
        """Test loading the model."""
        # Mock components
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_pipeline = MagicMock()
        mock_hf_pipeline.return_value = mock_pipeline

        # Force langchain_available and transformers_available to be True for this test
        with (
            patch("src.fine_tuning.fine_tuner.langchain_available", True),
            patch("src.fine_tuning.fine_tuner.transformers_available", True),
        ):
            tuner = FineTuner(model_name="distilgpt2")
            tuner._load_model_if_needed()
            self.assertTrue(tuner.model_loaded)
            mock_tokenizer.from_pretrained.assert_called_once()
            mock_model.from_pretrained.assert_called_once()

    @patch("src.fine_tuning.fine_tuner.HuggingFacePipeline")
    @patch("src.fine_tuning.fine_tuner.LLMChain")
    def test_process_query(self, mock_chain, mock_hf_pipeline):
        """Test processing a query."""
        # Mock components
        mock_pipeline = MagicMock()
        mock_hf_pipeline.return_value = mock_pipeline

        mock_chain_instance = MagicMock()
        mock_chain.return_value = mock_chain_instance
        mock_chain_instance.invoke.return_value = {
            "text": "The revenue was $10 million."
        }

        # Force langchain_available and transformers_available to be True for this test
        with (
            patch("src.fine_tuning.fine_tuner.langchain_available", True),
            patch("src.fine_tuning.fine_tuner.transformers_available", True),
        ):
            tuner = FineTuner(model_name="distilgpt2")
            tuner.chain = mock_chain_instance

            result = tuner.process_query("What was the revenue?")
            self.assertIsInstance(result, dict)
            self.assertIn("query", result)
            self.assertIn("answer", result)
            self.assertIn("confidence", result)
            self.assertIn("response_time", result)

            mock_chain_instance.invoke.assert_called_once()

    @patch("src.fine_tuning.fine_tuner.HuggingFacePipeline")
    @patch("src.fine_tuning.fine_tuner.AutoModelForCausalLM")
    @patch("src.fine_tuning.fine_tuner.AutoTokenizer")
    @patch("src.fine_tuning.fine_tuner.Trainer")
    @patch("src.fine_tuning.fine_tuner.TrainingArguments")
    def test_fine_tune(
        self, mock_args, mock_trainer, mock_model, mock_tokenizer, mock_hf_pipeline
    ):
        """Test fine-tuning the model."""
        # Mock components
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_args_instance = MagicMock()
        mock_args.return_value = mock_args_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Force langchain_available and transformers_available to be True for this test
        with (
            patch("src.fine_tuning.fine_tuner.langchain_available", True),
            patch("src.fine_tuning.fine_tuner.transformers_available", True),
            patch("src.fine_tuning.fine_tuner.datasets_available", True),
        ):
            tuner = FineTuner(model_name="distilgpt2")

            # Fine-tune the model
            tuner.fine_tune(
                qa_file=self.data_dir / "qa_pairs.json",
                output_dir=self.model_dir,
                num_train_epochs=1,
            )

            mock_tokenizer.from_pretrained.assert_called_once()
            mock_model.from_pretrained.assert_called_once()
            mock_args.assert_called_once()
            mock_trainer.assert_called_once()
            mock_trainer_instance.train.assert_called_once()

    @patch("src.fine_tuning.fine_tuner.HuggingFacePipeline")
    @patch("src.fine_tuning.fine_tuner.AutoModelForCausalLM")
    @patch("src.fine_tuning.fine_tuner.AutoTokenizer")
    def test_save_load(self, mock_tokenizer, mock_model, mock_hf_pipeline):
        """Test saving and loading the model."""
        # Mock components
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Force langchain_available and transformers_available to be True for this test
        with (
            patch("src.fine_tuning.fine_tuner.langchain_available", True),
            patch("src.fine_tuning.fine_tuner.transformers_available", True),
        ):
            tuner = FineTuner(model_name="distilgpt2")
            tuner.model = mock_model_instance
            tuner.tokenizer = mock_tokenizer_instance

            # Save model
            tuner.save(self.model_dir)
            mock_model_instance.save_pretrained.assert_called_once()
            mock_tokenizer_instance.save_pretrained.assert_called_once()

            # Load model
            new_tuner = FineTuner(model_name=None)
            new_tuner.load(self.model_dir)
            mock_tokenizer.from_pretrained.assert_called_with(self.model_dir)
            mock_model.from_pretrained.assert_called_with(self.model_dir)

    def test_quick_fine_tune(self):
        """Test the quick fine-tuning functionality."""
        qa_pairs = [
            {
                "question": "What was Apple's revenue?",
                "answer": "Apple reported $383.3 billion in revenue.",
            },
            {
                "question": "How did services perform?",
                "answer": "Services grew 8.2% year-over-year.",
            },
        ]

        with (
            patch("src.fine_tuning.fine_tuner.transformers_available", True),
            patch("src.fine_tuning.fine_tuner.datasets_available", True),
            patch("src.fine_tuning.fine_tuner.Dataset") as mock_dataset,
            patch("src.fine_tuning.fine_tuner.TrainingArguments") as mock_args,
            patch("src.fine_tuning.fine_tuner.Trainer") as mock_trainer,
            patch(
                "src.fine_tuning.fine_tuner.DataCollatorForLanguageModeling"
            ) as mock_collator,
        ):
            # Mock dataset operations
            mock_dataset_instance = MagicMock()
            mock_dataset.from_list.return_value = mock_dataset_instance
            mock_dataset_instance.map.return_value = mock_dataset_instance

            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            tuner = FineTuner(model_name="distilgpt2")
            tuner.tokenizer = MagicMock()
            tuner.model = MagicMock()

            # Test quick fine-tuning
            result = tuner.quick_fine_tune(qa_pairs)
            self.assertTrue(result)
            mock_trainer_instance.train.assert_called_once()

    def test_fallback_functionality(self):
        """Test fallback functionality when transformers is not available."""
        # Force transformers_available to be False for this test
        with patch("src.fine_tuning.fine_tuner.transformers_available", False):
            tuner = FineTuner(model_name="distilgpt2")
            result = tuner.process_query("What was the revenue?")
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
