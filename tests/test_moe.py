"""
Test script for the Mixture of Experts fine-tuning system.
"""

import json
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import the MoE fine-tuner
from src.fine_tuning.mixture_of_experts import (
    MixtureOfExpertsFineTuner,
    create_moe_fine_tuner,
)


def load_qa_pairs(qa_pairs_dir):
    """Load Q&A pairs from the data directory."""
    qa_pairs_dir = Path(qa_pairs_dir)

    # Load training data
    train_file = qa_pairs_dir / "financial_qa_train.json"
    with open(train_file, "r", encoding="utf-8") as f:
        train_pairs = json.load(f)

    # Load test data
    test_file = qa_pairs_dir / "financial_qa_test.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    # Load official questions
    official_file = qa_pairs_dir / "official_questions.json"
    with open(official_file, "r", encoding="utf-8") as f:
        official_questions = json.load(f)

    return train_pairs, test_pairs, official_questions


def enrich_qa_pairs_with_sections(qa_pairs):
    """Enrich Q&A pairs with section information based on question content."""
    enriched_pairs = []

    # Define section keywords for classification
    section_keywords = {
        "income_statement": [
            "revenue",
            "sales",
            "income",
            "profit",
            "earnings",
            "ebitda",
            "margin",
            "cost",
            "expense",
            "operating",
            "gross",
            "net income",
            "eps",
        ],
        "balance_sheet": [
            "assets",
            "liabilities",
            "equity",
            "debt",
            "cash",
            "inventory",
            "receivables",
            "payables",
            "balance sheet",
            "book value",
        ],
        "cash_flow": [
            "cash flow",
            "operating cash",
            "free cash",
            "capex",
            "capital expenditure",
            "financing",
            "investing",
            "dividends",
            "share repurchase",
        ],
        "notes_mda": [
            "guidance",
            "outlook",
            "strategy",
            "risk",
            "management",
            "discussion",
            "analysis",
            "segment",
            "geographic",
            "employees",
            "environmental",
        ],
    }

    for pair in qa_pairs:
        question = pair["question"].lower()

        # Determine section based on keywords
        section_scores = {}
        for section, keywords in section_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question)
            section_scores[section] = score

        # Assign section with highest score, default to income_statement
        section = (
            max(section_scores, key=section_scores.get)
            if any(section_scores.values())
            else "income_statement"
        )

        # Create enriched pair
        enriched_pair = pair.copy()
        enriched_pair["section"] = section
        enriched_pairs.append(enriched_pair)

    return enriched_pairs


def test_moe_system():
    """Test the Mixture of Experts fine-tuning system."""
    # Define paths
    project_root = Path.cwd()
    data_dir = project_root / "data"
    qa_pairs_dir = data_dir / "qa_pairs"
    moe_model_dir = project_root / "models" / "moe_fine_tuned"

    # Create model directory if it doesn't exist
    moe_model_dir.mkdir(parents=True, exist_ok=True)

    # Load Q&A pairs
    logger.info("Loading Q&A pairs...")
    try:
        train_pairs, test_pairs, official_questions = load_qa_pairs(qa_pairs_dir)
        logger.info(
            f" Loaded {len(train_pairs)} training pairs, {len(test_pairs)} test pairs"
        )
    except Exception as e:
        logger.error(f"Failed to load Q&A pairs: {e}")
        return

    # Enrich Q&A pairs with section information
    logger.info("Enriching Q&A pairs with section information...")
    enriched_train_pairs = enrich_qa_pairs_with_sections(train_pairs)

    # Initialize the MoE fine-tuner
    logger.info("Initializing MoE fine-tuner...")
    moe_tuner = create_moe_fine_tuner(
        model_name="distilgpt2", output_dir=str(moe_model_dir)
    )

    # Train the MoE system
    logger.info("Training MoE system...")
    start_time = time.time()
    success = moe_tuner.train_moe(
        enriched_train_pairs, epochs=3
    )  # Increased to 3 epochs
    train_time = time.time() - start_time

    if success:
        logger.info(f" MoE training completed in {train_time:.2f} seconds")

        # Get expert statistics
        expert_stats = moe_tuner.get_expert_stats()
        logger.info(f"Expert stats: {json.dumps(expert_stats, indent=2)}")

        # Test with official questions
        logger.info("\nTesting with official questions:")
        for i, q_data in enumerate(official_questions):
            question = q_data["question"]
            ground_truth = q_data["answer"]
            question_type = q_data["type"]

            logger.info(f"\nQuestion {i + 1} ({question_type}):")
            logger.info(f"Q: {question}")
            logger.info(f"Ground truth: {ground_truth}")

            # Process the query using the MoE system
            result = moe_tuner.process_query(question)

            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Selected Expert: {result['selected_expert']}")
            logger.info(f"Expert Weights: {result['expert_weights']}")
            logger.info(f"Confidence: {result['confidence']:.3f}")
            logger.info(f"Response time: {result['response_time']:.3f}s")
    else:
        logger.error(" MoE training failed")


if __name__ == "__main__":
    test_moe_system()
