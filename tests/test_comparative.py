#!/usr/bin/env python
"""
Comparative Analysis: MoE vs Standard Fine-Tuning.

This script compares the performance of the Mixture of Experts approach
with standard fine-tuning on financial Q&A tasks.
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
from src.fine_tuning.mixture_of_experts import create_moe_fine_tuner
from src.fine_tuning.fine_tuner import FineTuner

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


def calculate_similarity(answer1, answer2):
    """Calculate text similarity between two answers."""
    return SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()


def evaluate_model(model, test_questions, ground_truths):
    """Evaluate a model on test questions."""
    results = []

    for i, question in enumerate(test_questions):
        ground_truth = ground_truths[i]

        # Process query
        start_time = time.time()
        result = model.process_query(question)
        response_time = time.time() - start_time

        # Get answer
        answer = result.get("answer", "")

        # Calculate similarity
        similarity = calculate_similarity(answer, ground_truth)

        # Determine if correct (similarity > 0.5)
        is_correct = similarity > 0.5

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "similarity": similarity,
                "is_correct": is_correct,
                "response_time": response_time,
            }
        )

    # Calculate metrics
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    avg_similarity = sum(r["similarity"] for r in results) / len(results)
    avg_response_time = sum(r["response_time"] for r in results) / len(results)

    return {
        "accuracy": accuracy,
        "avg_similarity": avg_similarity,
        "avg_response_time": avg_response_time,
        "results": results,
    }


def run_comparative_analysis():
    """Run comparative analysis between MoE and standard fine-tuning."""
    # Define paths
    data_dir = project_root / "data"
    qa_pairs_dir = data_dir / "qa_pairs"
    moe_model_dir = project_root / "models" / "moe_fine_tuned"
    ft_model_dir = project_root / "models" / "fine_tuned"

    # Create model directories if they don't exist
    moe_model_dir.mkdir(parents=True, exist_ok=True)
    ft_model_dir.mkdir(parents=True, exist_ok=True)

    # Load Q&A pairs
    logger.info("Loading Q&A pairs...")
    train_pairs, test_pairs, official_questions = load_qa_pairs(qa_pairs_dir)
    logger.info(
        f"Loaded {len(train_pairs)} training pairs, {len(test_pairs)} test pairs"
    )

    # Enrich Q&A pairs with section information
    logger.info("Enriching Q&A pairs with section information...")
    enriched_train_pairs = enrich_qa_pairs_with_sections(train_pairs)

    # Initialize models
    logger.info("Initializing models...")

    # Initialize MoE model
    moe_model = create_moe_fine_tuner(
        model_name="distilgpt2", output_dir=str(moe_model_dir)
    )

    # Initialize standard fine-tuned model
    ft_model = FineTuner(
        model_name="distilgpt2",
        output_dir=str(ft_model_dir),
        use_peft=True,  # Use Parameter-Efficient Fine-Tuning (LoRA)
    )

    # Train MoE model
    logger.info("Training MoE model...")
    moe_start_time = time.time()
    moe_success = moe_model.train_moe(enriched_train_pairs, epochs=3)
    moe_train_time = time.time() - moe_start_time

    if not moe_success:
        logger.error("MoE training failed")
        return

    logger.info(f"MoE model trained in {moe_train_time:.2f} seconds")

    # Train standard fine-tuned model
    logger.info("Training standard fine-tuned model...")
    ft_start_time = time.time()
    try:
        ft_model.fine_tune(train_pairs)
        ft_train_time = time.time() - ft_start_time
        logger.info(f"Standard model trained in {ft_train_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Standard model training failed: {e}")
        return

    # Prepare evaluation data
    logger.info("Preparing evaluation data...")

    # Extract test questions and ground truths
    test_questions = [pair["question"] for pair in test_pairs]
    test_ground_truths = [pair["answer"] for pair in test_pairs]

    # Extract official questions and ground truths
    official_test_questions = [q["question"] for q in official_questions]
    official_ground_truths = [q["answer"] for q in official_questions]

    # Evaluate on test set
    logger.info("Evaluating models on test set...")

    # Evaluate MoE model
    moe_test_metrics = evaluate_model(moe_model, test_questions, test_ground_truths)

    # Evaluate standard model
    ft_test_metrics = evaluate_model(ft_model, test_questions, test_ground_truths)

    # Evaluate on official questions
    logger.info("Evaluating models on official questions...")

    # Evaluate MoE model
    moe_official_metrics = evaluate_model(
        moe_model, official_test_questions, official_ground_truths
    )

    # Evaluate standard model
    ft_official_metrics = evaluate_model(
        ft_model, official_test_questions, official_ground_truths
    )

    # Display results
    logger.info("\n===== COMPARATIVE ANALYSIS RESULTS =====")

    # Training time comparison
    logger.info("\nTraining Time:")
    logger.info(f"MoE model: {moe_train_time:.2f} seconds")
    logger.info(f"Standard model: {ft_train_time:.2f} seconds")

    # Test set results
    logger.info("\nTest Set Results:")
    logger.info(f"MoE Accuracy: {moe_test_metrics['accuracy']:.2%}")
    logger.info(f"Standard Accuracy: {ft_test_metrics['accuracy']:.2%}")
    logger.info(f"MoE Avg Similarity: {moe_test_metrics['avg_similarity']:.2f}")
    logger.info(f"Standard Avg Similarity: {ft_test_metrics['avg_similarity']:.2f}")
    logger.info(f"MoE Avg Response Time: {moe_test_metrics['avg_response_time']:.3f}s")
    logger.info(
        f"Standard Avg Response Time: {ft_test_metrics['avg_response_time']:.3f}s"
    )

    # Official questions results
    logger.info("\nOfficial Questions Results:")
    logger.info(f"MoE Accuracy: {moe_official_metrics['accuracy']:.2%}")
    logger.info(f"Standard Accuracy: {ft_official_metrics['accuracy']:.2%}")
    logger.info(f"MoE Avg Similarity: {moe_official_metrics['avg_similarity']:.2f}")
    logger.info(f"Standard Avg Similarity: {ft_official_metrics['avg_similarity']:.2f}")
    logger.info(
        f"MoE Avg Response Time: {moe_official_metrics['avg_response_time']:.3f}s"
    )
    logger.info(
        f"Standard Avg Response Time: {ft_official_metrics['avg_response_time']:.3f}s"
    )

    # Sample comparison
    logger.info("\nSample Comparison (First Test Question):")
    logger.info(f"Question: {test_questions[0]}")
    logger.info(f"Ground Truth: {test_ground_truths[0]}")
    logger.info(f"MoE Answer: {moe_test_metrics['results'][0]['answer']}")
    logger.info(f"Standard Answer: {ft_test_metrics['results'][0]['answer']}")

    # Create comparison charts
    create_comparison_charts(
        moe_test_metrics,
        ft_test_metrics,
        moe_official_metrics,
        ft_official_metrics,
        moe_train_time,
        ft_train_time,
    )


def create_comparison_charts(
    moe_test, ft_test, moe_official, ft_official, moe_time, ft_time
):
    """Create comparison charts between MoE and standard fine-tuning."""
    try:
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Accuracy comparison (test set)
        axes[0, 0].bar(
            ["MoE", "Standard"],
            [moe_test["accuracy"], ft_test["accuracy"]],
            color=["#3498db", "#e74c3c"],
        )
        axes[0, 0].set_title("Test Set Accuracy")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_ylim(0, 1)

        # Similarity comparison (test set)
        axes[0, 1].bar(
            ["MoE", "Standard"],
            [moe_test["avg_similarity"], ft_test["avg_similarity"]],
            color=["#3498db", "#e74c3c"],
        )
        axes[0, 1].set_title("Test Set Avg Similarity")
        axes[0, 1].set_ylabel("Similarity")
        axes[0, 1].set_ylim(0, 1)

        # Response time comparison (test set)
        axes[0, 2].bar(
            ["MoE", "Standard"],
            [moe_test["avg_response_time"], ft_test["avg_response_time"]],
            color=["#3498db", "#e74c3c"],
        )
        axes[0, 2].set_title("Test Set Avg Response Time")
        axes[0, 2].set_ylabel("Time (seconds)")

        # Accuracy comparison (official questions)
        axes[1, 0].bar(
            ["MoE", "Standard"],
            [moe_official["accuracy"], ft_official["accuracy"]],
            color=["#3498db", "#e74c3c"],
        )
        axes[1, 0].set_title("Official Questions Accuracy")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_ylim(0, 1)

        # Similarity comparison (official questions)
        axes[1, 1].bar(
            ["MoE", "Standard"],
            [moe_official["avg_similarity"], ft_official["avg_similarity"]],
            color=["#3498db", "#e74c3c"],
        )
        axes[1, 1].set_title("Official Questions Avg Similarity")
        axes[1, 1].set_ylabel("Similarity")
        axes[1, 1].set_ylim(0, 1)

        # Training time comparison
        axes[1, 2].bar(
            ["MoE", "Standard"], [moe_time, ft_time], color=["#3498db", "#e74c3c"]
        )
        axes[1, 2].set_title("Training Time")
        axes[1, 2].set_ylabel("Time (seconds)")

        plt.tight_layout()
        plt.savefig("comparative_analysis.png")
        logger.info("Comparison charts saved to comparative_analysis.png")

    except Exception as e:
        logger.error(f"Failed to create comparison charts: {e}")


if __name__ == "__main__":
    run_comparative_analysis()
