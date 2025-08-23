"""
Tests for the MixtureOfExpertsFineTuner class (Group 118 Advanced Technique).
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

# Mock PEFT modules before importing the actual module
peft_mock = MagicMock()
get_peft_model_mock = MagicMock()
get_peft_model_mock.return_value = MagicMock()
lora_config_mock = MagicMock()
task_type_mock = MagicMock()
peft_model_mock = MagicMock()

# Apply mocks
import sys

sys.modules["peft"] = peft_mock
sys.modules["peft.LoraConfig"] = lora_config_mock
sys.modules["peft.get_peft_model"] = get_peft_model_mock
sys.modules["peft.TaskType"] = task_type_mock
sys.modules["peft.PeftModel"] = peft_model_mock

# Now import the module under test
from src.fine_tuning.mixture_of_experts import (
    MixtureOfExpertsFineTuner,
    FinancialSectionRouter,
)


class TestFinancialSectionRouter(unittest.TestCase):
    """Test cases for the FinancialSectionRouter class."""

    def setUp(self):
        """Set up test environment."""
        self.router = FinancialSectionRouter()

        # Sample Q&A pairs for testing
        self.sample_qa_pairs = [
            {
                "question": "What was the total revenue for fiscal 2023?",
                "answer": "$383.3 billion",
                "section": "income_statement",
            },
            {
                "question": "How much cash and cash equivalents did the company have?",
                "answer": "$29.5 billion",
                "section": "balance_sheet",
            },
            {
                "question": "What were the operating cash flows?",
                "answer": "$110.5 billion",
                "section": "cash_flow",
            },
            {
                "question": "What is the company's accounting policy for revenue recognition?",
                "answer": "Revenue is recognized when control transfers to the customer.",
                "section": "notes_mda",
            },
        ]

    def test_initialization(self):
        """Test FinancialSectionRouter initialization."""
        router = FinancialSectionRouter()
        self.assertIsNotNone(router)
        self.assertEqual(len(router.section_keywords), 4)
        self.assertIn("income_statement", router.section_keywords)
        self.assertIn("balance_sheet", router.section_keywords)
        self.assertIn("cash_flow", router.section_keywords)
        self.assertIn("notes_mda", router.section_keywords)

    def test_train_router(self):
        """Test router training functionality."""
        # Train the router
        self.router.train_router(self.sample_qa_pairs)

        # Check that models are trained
        self.assertIsNotNone(self.router.vectorizer)
        self.assertIsNotNone(self.router.classifier)
        self.assertIsNotNone(self.router.label_encoder)

        # Check that it can make predictions
        test_question = "What was the net income?"
        weights = self.router.route_question(test_question)

        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 4)

        # Weights should sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)

    def test_route_question_before_training(self):
        """Test routing before training (should use keyword fallback)."""
        test_question = "What was the total revenue?"
        weights = self.router.route_question(test_question)

        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 4)

        # Should use keyword-based routing
        # Revenue question should favor income_statement
        self.assertGreater(weights["income_statement"], 0)

    def test_keyword_based_routing(self):
        """Test keyword-based routing fallback."""
        test_cases = [
            ("revenue", "income_statement"),
            ("cash", "balance_sheet"),
            ("operating cash flow", "cash_flow"),
            ("management discussion", "notes_mda"),
        ]

        for question, expected_section in test_cases:
            weights = self.router.route_question(question)

            # The expected section should have the highest weight
            max_section = max(weights.keys(), key=lambda k: weights[k])
            self.assertEqual(max_section, expected_section)

    def test_route_question_after_training(self):
        """Test routing after training with classifier."""
        # Train the router
        self.router.train_router(self.sample_qa_pairs)

        test_cases = [
            "What was the total revenue for the year?",
            "How much cash did the company have?",
            "What were the cash flows from operations?",
            "What is the revenue recognition policy?",
        ]

        for question in test_cases:
            weights = self.router.route_question(question)

            self.assertIsInstance(weights, dict)
            self.assertEqual(len(weights), 4)

            # All weights should be non-negative
            for weight in weights.values():
                self.assertGreaterEqual(weight, 0)

            # Weights should sum to approximately 1
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=2)


class TestMixtureOfExpertsFineTuner(unittest.TestCase):
    """Test cases for the MixtureOfExpertsFineTuner class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "model"
        self.model_dir.mkdir(exist_ok=True)

        # Sample Q&A pairs for testing
        self.sample_qa_pairs = [
            {
                "question": "What was the total revenue for fiscal 2023?",
                "answer": "$383.3 billion",
                "section": "income_statement",
            },
            {
                "question": "How much cash and cash equivalents did the company have?",
                "answer": "$29.5 billion",
                "section": "balance_sheet",
            },
            {
                "question": "What were the operating cash flows?",
                "answer": "$110.5 billion",
                "section": "cash_flow",
            },
            {
                "question": "What is the company's accounting policy for revenue recognition?",
                "answer": "Revenue is recognized when control transfers to the customer.",
                "section": "notes_mda",
            },
        ]

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_initialization(self):
        """Test MixtureOfExpertsFineTuner initialization."""
        with patch("src.fine_tuning.mixture_of_experts.AutoModelForCausalLM"):
            with patch("src.fine_tuning.mixture_of_experts.AutoTokenizer"):
                moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

                self.assertIsNotNone(moe)
                self.assertEqual(str(moe.output_dir), str(self.model_dir))
                self.assertIsNotNone(moe.router)
                self.assertEqual(len(moe.expert_sections), 4)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", False)
    def test_initialization_without_torch(self):
        """Test initialization when PyTorch is not available."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        self.assertIsNotNone(moe)
        self.assertIsNone(moe.base_model)
        self.assertIsNone(moe.tokenizer)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    @patch("src.fine_tuning.mixture_of_experts.get_peft_model")
    @patch("src.fine_tuning.mixture_of_experts.LoraConfig")
    def test_create_experts(self, mock_lora_config, mock_get_peft_model):
        """Test expert creation functionality."""
        # Mock the PEFT components
        mock_lora_config_instance = MagicMock()
        mock_lora_config.return_value = mock_lora_config_instance

        mock_expert_model = MagicMock()
        mock_get_peft_model.return_value = mock_expert_model

        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        # Mock the model and tokenizer to avoid loading actual models
        with patch.object(moe, "base_model", MagicMock()):
            with patch.object(moe, "tokenizer", MagicMock()):
                moe.create_experts()

                # Check that experts were created
                self.assertEqual(len(moe.experts), 4)

                # Check that PEFT was used
                self.assertEqual(mock_lora_config.call_count, 4)
                self.assertEqual(mock_get_peft_model.call_count, 4)

                # Check that each expert section has an expert
                for section in moe.expert_sections:
                    self.assertIn(section, moe.experts)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_train_moe_basic(self):
        """Test basic MoE training functionality."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        # Mock dependencies to avoid actual training
        with patch.object(moe, "_initialize_components"):
            with patch.object(moe, "create_experts"):
                with patch.object(moe, "_train_expert"):
                    with patch.object(moe.router, "train_router"):
                        # Set up mock experts
                        moe.experts = {
                            section: MagicMock() for section in moe.expert_sections
                        }

                        result = moe.train_moe(self.sample_qa_pairs, epochs=1)

                        self.assertIsInstance(result, bool)
                        self.assertTrue(result)

                        # Check that router was trained
                        moe.router.train_router.assert_called_once()

                        # Check that each expert was trained
                        self.assertEqual(moe._train_expert.call_count, 4)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_process_query_with_routing(self):
        """Test query processing with expert routing."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))
        moe.is_trained = True  # Set as trained

        # Mock the router and experts
        mock_weights = {
            "income_statement": 0.8,
            "balance_sheet": 0.1,
            "cash_flow": 0.05,
            "notes_mda": 0.05,
        }

        with patch.object(moe.router, "route_question", return_value=mock_weights):
            with patch.object(
                moe, "_generate_with_expert", return_value="Mocked answer"
            ):
                # Set up mock experts
                moe.experts = {"income_statement": MagicMock()}

                result = moe.process_query("What was the revenue?")

                self.assertIsInstance(result, dict)
                self.assertIn("answer", result)
                self.assertIn("expert_weights", result)
                self.assertIn("selected_expert", result)
                self.assertIn("moe_metadata", result)

                # Check that the highest weight expert was selected
                self.assertEqual(result["selected_expert"], "income_statement")
                self.assertEqual(result["answer"], "Mocked answer")

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_process_query_fallback(self):
        """Test query processing fallback when experts are not available."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))
        moe.is_trained = True  # Set as trained

        # No experts available
        moe.experts = {}

        # Mock the router
        mock_weights = {
            "income_statement": 0.8,
            "balance_sheet": 0.1,
            "cash_flow": 0.05,
            "notes_mda": 0.05,
        }

        with patch.object(moe.router, "route_question", return_value=mock_weights):
            with patch.object(
                moe, "_generate_fallback_response", return_value="Fallback answer"
            ):
                result = moe.process_query("What was the revenue?")

                self.assertIsInstance(result, dict)
                self.assertIn("answer", result)
                self.assertIn("expert_weights", result)
                self.assertIn("selected_expert", result)

                # Should use fallback
                self.assertEqual(result["answer"], "Fallback answer")

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_expert_weight_distribution(self):
        """Test that expert weights are properly distributed."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        # Train router with sample data
        with patch.object(moe.router, "vectorizer", MagicMock()):
            with patch.object(moe.router, "classifier", MagicMock()):
                with patch.object(moe.router, "label_encoder", MagicMock()):
                    moe.router.train_router(self.sample_qa_pairs)

        test_questions = [
            "What was the total revenue?",
            "How much cash did the company have?",
            "What were the operating cash flows?",
            "What is the accounting policy?",
        ]

        for question in test_questions:
            weights = moe.router.route_question(question)

            # Weights should sum to 1
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=2)

            # At least one expert should have significant weight
            max_weight = max(weights.values())
            self.assertGreater(max_weight, 0.1)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_generate_fallback_response(self):
        """Test fallback response generation."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        expert_weights = {
            "income_statement": 0.8,
            "balance_sheet": 0.2,
            "cash_flow": 0.0,
            "notes_mda": 0.0,
        }
        response = moe._generate_fallback_response(
            "What was the revenue?", expert_weights
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIn("financial", response.lower())

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_section_grouping(self):
        """Test that Q&A pairs are correctly grouped by section."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        # Mock training to test grouping logic
        mock_experts = {section: MagicMock() for section in moe.expert_sections}
        with patch.object(moe, "_initialize_components"):
            with patch.object(moe, "create_experts", return_value=mock_experts):
                with patch.object(moe, "_train_expert") as mock_train:
                    with patch.object(moe.router, "train_router"):
                        moe.experts = mock_experts  # Set the experts
                        moe.train_moe(self.sample_qa_pairs, epochs=1)

                        # Check that _train_expert was called for sections with data
                        self.assertGreater(mock_train.call_count, 0)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_lora_config_parameters(self):
        """Test LoRA configuration parameters."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))

        # Check LoRA configuration
        self.assertEqual(moe.lora_rank, 8)  # Default value
        self.assertEqual(moe.lora_alpha, 32)
        self.assertEqual(moe.lora_dropout, 0.1)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_metadata_tracking(self):
        """Test that MoE metadata is properly tracked."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))
        moe.is_trained = True  # Set as trained

        mock_weights = {
            "income_statement": 0.7,
            "balance_sheet": 0.2,
            "cash_flow": 0.05,
            "notes_mda": 0.05,
        }

        with patch.object(moe.router, "route_question", return_value=mock_weights):
            with patch.object(moe, "_generate_with_expert", return_value="Test answer"):
                moe.experts = {"income_statement": MagicMock()}

                result = moe.process_query("Test question")

                metadata = result["moe_metadata"]
                self.assertIn("status", metadata)
                self.assertEqual(metadata["status"], "success")
                self.assertIn("num_experts", metadata)
                self.assertIn("routing_method", metadata)

    @patch("src.fine_tuning.mixture_of_experts.torch_available", True)
    @patch("src.fine_tuning.mixture_of_experts.peft_available", True)
    def test_expert_selection_logic(self):
        """Test expert selection logic with different weight distributions."""
        moe = MixtureOfExpertsFineTuner(output_dir=str(self.model_dir))
        moe.is_trained = True  # Set as trained

        test_cases = [
            # Clear winner
            {
                "income_statement": 0.8,
                "balance_sheet": 0.1,
                "cash_flow": 0.05,
                "notes_mda": 0.05,
            },
            # Close competition
            {
                "income_statement": 0.4,
                "balance_sheet": 0.35,
                "cash_flow": 0.15,
                "notes_mda": 0.1,
            },
            # Even distribution
            {
                "income_statement": 0.25,
                "balance_sheet": 0.25,
                "cash_flow": 0.25,
                "notes_mda": 0.25,
            },
        ]

        for weights in test_cases:
            with patch.object(moe.router, "route_question", return_value=weights):
                with patch.object(moe, "_generate_with_expert", return_value="Test"):
                    # Set up experts
                    expected_expert = max(weights.keys(), key=weights.get)
                    moe.experts = {expected_expert: MagicMock()}

                    result = moe.process_query("Test question")

                    # Should select the expert with highest weight
                    self.assertEqual(result["selected_expert"], expected_expert)


if __name__ == "__main__":
    unittest.main()
