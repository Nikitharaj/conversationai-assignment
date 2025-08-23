#!/usr/bin/env python
"""
Router Evaluation for Financial Q&A MoE System.

This script evaluates the performance of the router component in the MoE system.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    FinancialSectionRouter,
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

    return train_pairs


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


def create_test_questions():
    """Create test questions for router evaluation."""
    test_questions = [
        # Income Statement Questions
        {
            "question": "What was the revenue in 2023?",
            "expected_section": "income_statement",
        },
        {
            "question": "How much profit did the company make?",
            "expected_section": "income_statement",
        },
        {
            "question": "What were the operating expenses last year?",
            "expected_section": "income_statement",
        },
        {
            "question": "What was the EBITDA for the most recent quarter?",
            "expected_section": "income_statement",
        },
        {
            "question": "How did the gross margin change from 2022 to 2023?",
            "expected_section": "income_statement",
        },
        # Balance Sheet Questions
        {"question": "What are the total assets?", "expected_section": "balance_sheet"},
        {
            "question": "How much debt does the company have?",
            "expected_section": "balance_sheet",
        },
        {
            "question": "What is the current cash position?",
            "expected_section": "balance_sheet",
        },
        {
            "question": "What was the change in inventory from 2022 to 2023?",
            "expected_section": "balance_sheet",
        },
        {
            "question": "What is the shareholders' equity?",
            "expected_section": "balance_sheet",
        },
        # Cash Flow Questions
        {
            "question": "How much operating cash flow was generated?",
            "expected_section": "cash_flow",
        },
        {
            "question": "What were the capital expenditures in 2023?",
            "expected_section": "cash_flow",
        },
        {
            "question": "How much was spent on dividends?",
            "expected_section": "cash_flow",
        },
        {
            "question": "What was the free cash flow in the most recent year?",
            "expected_section": "cash_flow",
        },
        {
            "question": "How much cash was used for financing activities?",
            "expected_section": "cash_flow",
        },
        # Notes/MD&A Questions
        {
            "question": "What is the company's strategy for 2024?",
            "expected_section": "notes_mda",
        },
        {
            "question": "What are the main risk factors for the business?",
            "expected_section": "notes_mda",
        },
        {
            "question": "How does management view the competitive landscape?",
            "expected_section": "notes_mda",
        },
        {
            "question": "What are the environmental initiatives mentioned?",
            "expected_section": "notes_mda",
        },
        {
            "question": "What is the outlook for the next fiscal year?",
            "expected_section": "notes_mda",
        },
        # Ambiguous Questions
        {
            "question": "How is the company performing?",
            "expected_section": "income_statement",
        },
        {
            "question": "What are the financial highlights?",
            "expected_section": "income_statement",
        },
        {
            "question": "How much money does the company have?",
            "expected_section": "balance_sheet",
        },
        {"question": "Is the company growing?", "expected_section": "notes_mda"},
        {
            "question": "What are the key metrics to watch?",
            "expected_section": "notes_mda",
        },
        # Irrelevant Questions
        {
            "question": "What is your favorite color?",
            "expected_section": "income_statement",
        },
        {
            "question": "How is the weather today?",
            "expected_section": "income_statement",
        },
        {"question": "Who is the president?", "expected_section": "income_statement"},
        {"question": "What time is it?", "expected_section": "income_statement"},
        {"question": "Can you tell me a joke?", "expected_section": "income_statement"},
    ]

    return test_questions


def test_router_component():
    """Test the router component of the MoE system."""
    # Define paths
    data_dir = project_root / "data"
    qa_pairs_dir = data_dir / "qa_pairs"

    # Load Q&A pairs
    logger.info("Loading Q&A pairs...")
    train_pairs = load_qa_pairs(qa_pairs_dir)
    logger.info(f"Loaded {len(train_pairs)} training pairs")

    # Enrich Q&A pairs with section information
    logger.info("Enriching Q&A pairs with section information...")
    enriched_train_pairs = enrich_qa_pairs_with_sections(train_pairs)

    # Initialize router
    logger.info("Initializing router...")
    router = FinancialSectionRouter()

    # Train router
    logger.info("Training router...")
    router.train_router(enriched_train_pairs)

    # Create test questions
    test_questions = create_test_questions()

    # Test router
    logger.info("Testing router...")

    results = []
    section_names = ["income_statement", "balance_sheet", "cash_flow", "notes_mda"]

    for q in test_questions:
        question = q["question"]
        expected_section = q["expected_section"]

        # Route question
        expert_weights = router.route_question(question)
        selected_expert = max(expert_weights, key=expert_weights.get)
        confidence = expert_weights[selected_expert]

        # Store results
        result = {
            "question": question,
            "expected_section": expected_section,
            "selected_section": selected_expert,
            "confidence": confidence,
            "correct": expected_section == selected_expert,
            "weights": expert_weights,
        }

        results.append(result)

    # Calculate metrics
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)

    # Calculate section-specific metrics
    section_metrics = {}
    for section in section_names:
        section_questions = [r for r in results if r["expected_section"] == section]
        section_correct = sum(1 for r in section_questions if r["correct"])
        section_accuracy = (
            section_correct / len(section_questions) if section_questions else 0
        )
        section_avg_confidence = (
            sum(r["confidence"] for r in section_questions) / len(section_questions)
            if section_questions
            else 0
        )

        section_metrics[section] = {
            "accuracy": section_accuracy,
            "avg_confidence": section_avg_confidence,
            "count": len(section_questions),
        }

    # Display results
    logger.info("\n===== ROUTER EVALUATION RESULTS =====")
    logger.info(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")

    for section, metrics in section_metrics.items():
        logger.info(f"\n{section.upper()}:")
        logger.info(
            f"Accuracy: {metrics['accuracy']:.2%} ({int(metrics['accuracy'] * metrics['count'])}/{metrics['count']})"
        )
        logger.info(f"Average Confidence: {metrics['avg_confidence']:.2f}")

    # Create confusion matrix
    y_true = [r["expected_section"] for r in results]
    y_pred = [r["selected_section"] for r in results]

    cm = confusion_matrix(y_true, y_pred, labels=section_names)
    cm_df = pd.DataFrame(cm, index=section_names, columns=section_names)

    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm_df}")

    # Create visualizations
    create_router_visualizations(results, section_names, cm_df)

    # Sample routing analysis
    logger.info("\nSample Routing Analysis:")

    for i, result in enumerate(results[:5]):
        logger.info(f"\nQuestion {i + 1}: {result['question']}")
        logger.info(f"Expected Section: {result['expected_section']}")
        logger.info(f"Selected Section: {result['selected_section']}")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        logger.info(f"Correct: {result['correct']}")

        # Display weights
        logger.info("Section Weights:")
        for section, weight in result["weights"].items():
            logger.info(f"  - {section}: {weight:.2f}")


def create_router_visualizations(results, section_names, cm_df):
    """Create visualizations for router evaluation."""
    try:
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Accuracy by section
        section_accuracies = [
            sum(1 for r in results if r["expected_section"] == section and r["correct"])
            / sum(1 for r in results if r["expected_section"] == section)
            for section in section_names
        ]

        axes[0, 0].bar(section_names, section_accuracies, color="#3498db")
        axes[0, 0].set_title("Routing Accuracy by Section")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_ylim(0, 1)

        # Average confidence by section
        section_confidences = [
            sum(r["confidence"] for r in results if r["expected_section"] == section)
            / sum(1 for r in results if r["expected_section"] == section)
            for section in section_names
        ]

        axes[0, 1].bar(section_names, section_confidences, color="#2ecc71")
        axes[0, 1].set_title("Average Confidence by Section")
        axes[0, 1].set_ylabel("Confidence")
        axes[0, 1].set_ylim(0, 1)

        # Confusion matrix heatmap
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
        axes[1, 0].set_title("Confusion Matrix")
        axes[1, 0].set_xlabel("Predicted Section")
        axes[1, 0].set_ylabel("True Section")

        # Confidence distribution
        sns.histplot(
            [r["confidence"] for r in results],
            bins=10,
            kde=True,
            ax=axes[1, 1],
            color="#9b59b6",
        )
        axes[1, 1].set_title("Confidence Distribution")
        axes[1, 1].set_xlabel("Confidence")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig("router_evaluation.png")
        logger.info("Router evaluation visualization saved to router_evaluation.png")

    except Exception as e:
        logger.error(f"Failed to create router visualizations: {e}")


if __name__ == "__main__":
    test_router_component()
