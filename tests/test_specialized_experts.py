#!/usr/bin/env python
"""
Specialized Experts Test for Financial Q&A.

This script tests the performance of each specialized expert in the MoE system.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import project modules
from src.fine_tuning.mixture_of_experts import (
    MixtureOfExpertsFineTuner,
    create_moe_fine_tuner,
)

# Add the project root to the path if needed
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


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

    return train_pairs, test_pairs


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


def create_section_specific_test_questions():
    """Create test questions specific to each financial section."""
    section_questions = {
        "income_statement": [
            {
                "question": "What was the revenue in 2023?",
                "expected_answer": "The revenue was $1,250 million.",
            },
            {
                "question": "How much was the net income in the most recent year?",
                "expected_answer": "The net income was $187.5 million.",
            },
            {
                "question": "What was the gross profit margin?",
                "expected_answer": "The gross profit margin was 40%.",
            },
        ],
        "balance_sheet": [
            {
                "question": "What were the total assets in 2023?",
                "expected_answer": "The total assets were $1,500 million.",
            },
            {
                "question": "How much equity did the company have?",
                "expected_answer": "The total equity was $850 million.",
            },
            {
                "question": "What was the debt-to-equity ratio?",
                "expected_answer": "The debt-to-equity ratio was 0.765.",
            },
        ],
        "cash_flow": [
            {
                "question": "How much operating cash flow was generated in 2023?",
                "expected_answer": "The operating cash flow was $220 million.",
            },
            {
                "question": "What was the free cash flow in the most recent year?",
                "expected_answer": "The free cash flow was $50 million.",
            },
            {
                "question": "How much was spent on capital expenditures?",
                "expected_answer": "Capital expenditures were $170 million.",
            },
        ],
        "notes_mda": [
            {
                "question": "What is the company's strategy for growth?",
                "expected_answer": "The company's strategy focuses on innovation and expansion into new markets.",
            },
            {
                "question": "What is the outlook for 2024?",
                "expected_answer": "The company anticipates revenue growth of 8-10% and improved profit margins.",
            },
            {
                "question": "What are the main risk factors for the business?",
                "expected_answer": "The main risk factors are not explicitly mentioned in the provided documents.",
            },
        ],
    }

    return section_questions


def calculate_similarity(answer1, answer2):
    """Calculate text similarity between two answers."""
    return SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()


def test_specialized_experts():
    """Test the performance of each specialized expert in the MoE system."""
    # Define paths
    data_dir = project_root / "data"
    qa_pairs_dir = data_dir / "qa_pairs"
    moe_model_dir = project_root / "models" / "moe_fine_tuned"

    # Create model directory if it doesn't exist
    moe_model_dir.mkdir(parents=True, exist_ok=True)

    # Load Q&A pairs
    logger.info("Loading Q&A pairs...")
    train_pairs, test_pairs = load_qa_pairs(qa_pairs_dir)
    logger.info(
        f"Loaded {len(train_pairs)} training pairs, {len(test_pairs)} test pairs"
    )

    # Enrich Q&A pairs with section information
    logger.info("Enriching Q&A pairs with section information...")
    enriched_train_pairs = enrich_qa_pairs_with_sections(train_pairs)

    # Initialize the MoE model
    logger.info("Initializing MoE model...")
    moe_model = create_moe_fine_tuner(
        model_name="distilgpt2", output_dir=str(moe_model_dir)
    )

    # Train the MoE model
    logger.info("Training MoE model...")
    success = moe_model.train_moe(enriched_train_pairs, epochs=3)

    if not success:
        logger.error("MoE training failed")
        return

    # Get section-specific test questions
    section_questions = create_section_specific_test_questions()

    # Test each expert
    expert_results = {}

    for section, questions in section_questions.items():
        logger.info(f"\nTesting {section} expert...")
        section_results = []

        for q in questions:
            question = q["question"]
            expected_answer = q["expected_answer"]

            # Process query with MoE
            logger.info(f"Question: {question}")
            result = moe_model.process_query(question)

            # Get answer and expert info
            answer = result["answer"]
            selected_expert = result["selected_expert"]
            confidence = result["confidence"]

            # Calculate similarity
            similarity = calculate_similarity(answer, expected_answer)

            # Log results
            logger.info(f"Expected Expert: {section}")
            logger.info(f"Selected Expert: {selected_expert}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Answer: {answer}")
            logger.info(f"Expected Answer: {expected_answer}")
            logger.info(f"Similarity: {similarity:.2f}")

            # Store results
            section_results.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": answer,
                    "expected_expert": section,
                    "selected_expert": selected_expert,
                    "confidence": confidence,
                    "similarity": similarity,
                    "correct_routing": section == selected_expert,
                }
            )

        # Calculate section metrics
        correct_routing = sum(1 for r in section_results if r["correct_routing"])
        routing_accuracy = correct_routing / len(section_results)
        avg_similarity = sum(r["similarity"] for r in section_results) / len(
            section_results
        )
        avg_confidence = sum(r["confidence"] for r in section_results) / len(
            section_results
        )

        expert_results[section] = {
            "routing_accuracy": routing_accuracy,
            "avg_similarity": avg_similarity,
            "avg_confidence": avg_confidence,
            "results": section_results,
        }

    # Display summary
    logger.info("\n===== EXPERT EVALUATION SUMMARY =====")

    for section, metrics in expert_results.items():
        logger.info(f"\n{section.upper()} EXPERT:")
        logger.info(f"Routing Accuracy: {metrics['routing_accuracy']:.2%}")
        logger.info(f"Average Answer Similarity: {metrics['avg_similarity']:.2f}")
        logger.info(f"Average Confidence: {metrics['avg_confidence']:.2f}")

    # Calculate overall metrics
    all_results = []
    for section_metrics in expert_results.values():
        all_results.extend(section_metrics["results"])

    overall_routing_accuracy = sum(
        1 for r in all_results if r["correct_routing"]
    ) / len(all_results)
    overall_avg_similarity = sum(r["similarity"] for r in all_results) / len(
        all_results
    )

    logger.info("\nOVERALL METRICS:")
    logger.info(f"Overall Routing Accuracy: {overall_routing_accuracy:.2%}")
    logger.info(f"Overall Answer Similarity: {overall_avg_similarity:.2f}")

    # Create visualization
    create_expert_visualization(expert_results)


def create_expert_visualization(expert_results):
    """Create visualization of expert performance."""
    try:
        # Extract metrics
        sections = list(expert_results.keys())
        routing_accuracies = [expert_results[s]["routing_accuracy"] for s in sections]
        avg_similarities = [expert_results[s]["avg_similarity"] for s in sections]
        avg_confidences = [expert_results[s]["avg_confidence"] for s in sections]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Routing accuracy
        axes[0].bar(sections, routing_accuracies, color="#3498db")
        axes[0].set_title("Routing Accuracy by Section")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_ylim(0, 1)

        # Average similarity
        axes[1].bar(sections, avg_similarities, color="#2ecc71")
        axes[1].set_title("Average Answer Similarity by Section")
        axes[1].set_ylabel("Similarity")
        axes[1].set_ylim(0, 1)

        # Average confidence
        axes[2].bar(sections, avg_confidences, color="#e74c3c")
        axes[2].set_title("Average Confidence by Section")
        axes[2].set_ylabel("Confidence")
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("expert_evaluation.png")
        logger.info("Expert evaluation visualization saved to expert_evaluation.png")

    except Exception as e:
        logger.error(f"Failed to create expert visualization: {e}")


if __name__ == "__main__":
    test_specialized_experts()
