"""
LangChain-based evaluator for comparing RAG and Fine-Tuned models.

This module provides evaluation metrics for comparing RAG and Fine-Tuned models.
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import LangChain components
try:
    from langchain.evaluation import EvaluatorType
    from langchain.evaluation.schema import StringEvaluator
    from langchain_community.evaluation import (
        ExactMatchStringEvaluator,
        QAEvalChain,
        PairwiseStringEvaluator,
        load_evaluator,
    )
    from langchain_core.language_models import BaseLanguageModel
    from langchain_openai import ChatOpenAI

    langchain_available = True
except ImportError:
    # Only warn if this is not being imported during test execution
    import sys

    if not any("pytest" in arg for arg in sys.argv) and not any(
        "unittest" in arg for arg in sys.argv
    ):
        warnings.warn(
            "LangChain evaluation not available. Install with 'pip install langchain langchain-community langchain-openai'"
        )
    langchain_available = False


class Evaluator:
    """LangChain-based evaluator for comparing RAG and Fine-Tuned models."""

    def __init__(
        self,
        rag_system: Any = None,
        ft_model: Any = None,
        output_dir: Union[str, Path] = "./evaluation_results",
    ):
        """
        Initialize the evaluator.

        Args:
            rag_system: RAG system instance
            ft_model: Fine-tuned model instance
            output_dir: Directory to save evaluation results
        """
        self.rag_system = rag_system
        self.ft_model = ft_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators if LangChain is available
        self.exact_match_evaluator = None
        self.embedding_evaluator = None
        self.qa_eval_chain = None

        if langchain_available:
            try:
                # Initialize exact match evaluator
                self.exact_match_evaluator = ExactMatchStringEvaluator()

                # Initialize embedding evaluator
                self.embedding_evaluator = load_evaluator(
                    EvaluatorType.EMBEDDING_DISTANCE
                )

                # Initialize QA evaluator (requires OpenAI API key)
                if os.environ.get("OPENAI_API_KEY"):
                    self.qa_eval_chain = QAEvalChain.from_llm(
                        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
                    )
            except Exception as e:
                warnings.warn(f"Error initializing evaluators: {e}")

    def evaluate_test_set(self, test_file: Union[str, Path]):
        """
        Evaluate both models on a test set of Q&A pairs.

        Args:
            test_file: Path to the test file (JSON format)
        """
        test_file = Path(test_file)
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        # Load test data
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # Initialize results
        rag_results = []
        ft_results = []

        # Process each test example
        for i, example in enumerate(test_data):
            question = example["question"]
            ground_truth = example["answer"]
            question_type = example.get("type", "general")

            print(f"Processing example {i + 1}/{len(test_data)}: {question}")

            # RAG model
            if self.rag_system:
                try:
                    rag_result = self.rag_system.process_query(question)
                    rag_answer = rag_result["answer"]
                    rag_confidence = rag_result["confidence"]
                    rag_response_time = rag_result["response_time"]

                    # Evaluate correctness
                    is_correct = self._evaluate_answer(
                        rag_answer, ground_truth, question
                    )

                    # Store result
                    rag_results.append(
                        {
                            "question": question,
                            "answer": rag_answer,
                            "ground_truth": ground_truth,
                            "is_correct": is_correct,
                            "confidence": rag_confidence,
                            "response_time": rag_response_time,
                            "question_type": question_type,
                        }
                    )
                except Exception as e:
                    print(f"Error with RAG system: {e}")

            # Fine-tuned model
            if self.ft_model:
                try:
                    ft_result = self.ft_model.process_query(question)
                    ft_answer = ft_result["answer"]
                    ft_confidence = ft_result["confidence"]
                    ft_response_time = ft_result["response_time"]

                    # Evaluate correctness
                    is_correct = self._evaluate_answer(
                        ft_answer, ground_truth, question
                    )

                    # Store result
                    ft_results.append(
                        {
                            "question": question,
                            "answer": ft_answer,
                            "ground_truth": ground_truth,
                            "is_correct": is_correct,
                            "confidence": ft_confidence,
                            "response_time": ft_response_time,
                            "question_type": question_type,
                        }
                    )
                except Exception as e:
                    print(f"Error with Fine-Tuned model: {e}")

        # Calculate summary statistics
        rag_summary = self._calculate_summary(rag_results)
        ft_summary = self._calculate_summary(ft_results)

        # Save results
        self._save_results(rag_results, ft_results, rag_summary, ft_summary)

        # Create visualizations
        self._create_visualizations(rag_summary, ft_summary)

        return {
            "rag": rag_summary,
            "ft": ft_summary,
            "rag_results": rag_results,
            "ft_results": ft_results,
        }

    def _evaluate_answer(
        self, answer: str, ground_truth: str, question: str = ""
    ) -> bool:
        """
        Evaluate the correctness of an answer.

        Args:
            answer: The model's answer
            ground_truth: The ground truth answer
            question: The question (for context)

        Returns:
            bool: Whether the answer is correct
        """
        # Start with exact match
        if self.exact_match_evaluator:
            try:
                result = self.exact_match_evaluator.evaluate_strings(
                    prediction=answer, reference=ground_truth
                )
                if result["score"] == 1.0:
                    return True
            except Exception:
                pass

        # Try embedding distance if available
        if self.embedding_evaluator:
            try:
                result = self.embedding_evaluator.evaluate_strings(
                    prediction=answer, reference=ground_truth
                )
                if result["score"] >= 0.8:  # High semantic similarity
                    return True
            except Exception:
                pass

        # Try QA evaluation if available
        if self.qa_eval_chain and question:
            try:
                eval_result = self.qa_eval_chain.evaluate(
                    [{"question": question, "answer": ground_truth}],
                    [{"question": question, "answer": answer}],
                )
                if eval_result[0]["text"].lower().startswith("yes"):
                    return True
            except Exception:
                pass

        # Fallback to simple substring check
        answer_lower = answer.lower()
        ground_truth_lower = ground_truth.lower()

        # Check if key parts of the ground truth are in the answer
        key_parts = ground_truth_lower.split()
        if len(key_parts) > 3:
            # For longer answers, check if important words are present
            important_words = [
                word
                for word in key_parts
                if len(word) > 3 and word.isalnum() and not word.isdigit()
            ]
            matches = sum(1 for word in important_words if word in answer_lower)
            if matches / len(important_words) >= 0.7:  # 70% of important words match
                return True

        # Check for numeric values
        import re

        ground_truth_numbers = re.findall(r"\d+\.?\d*", ground_truth)
        answer_numbers = re.findall(r"\d+\.?\d*", answer)
        if ground_truth_numbers and set(ground_truth_numbers).issubset(
            set(answer_numbers)
        ):
            return True

        return False

    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics from results.

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary of summary statistics
        """
        if not results:
            return {
                "accuracy": 0.0,
                "avg_response_time": 0.0,
                "avg_confidence": 0.0,
            }

        # Basic metrics
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / len(results)
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        summary = {
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "avg_confidence": avg_confidence,
            "total_examples": len(results),
            "correct_count": correct_count,
        }

        # Calculate metrics by question type if available
        question_types = set(
            r["question_type"] for r in results if "question_type" in r
        )
        if question_types:
            for qtype in question_types:
                type_results = [r for r in results if r.get("question_type") == qtype]
                if type_results:
                    correct = sum(1 for r in type_results if r["is_correct"])
                    accuracy = correct / len(type_results)
                    avg_time = sum(r["response_time"] for r in type_results) / len(
                        type_results
                    )
                    avg_conf = sum(r["confidence"] for r in type_results) / len(
                        type_results
                    )

                    summary[f"{qtype}_accuracy"] = accuracy
                    summary[f"{qtype}_avg_time"] = avg_time
                    summary[f"{qtype}_avg_confidence"] = avg_conf
                    summary[f"{qtype}_count"] = len(type_results)

        return summary

    def _save_results(
        self,
        rag_results: List[Dict[str, Any]],
        ft_results: List[Dict[str, Any]],
        rag_summary: Dict[str, Any],
        ft_summary: Dict[str, Any],
    ):
        """
        Save evaluation results to files.

        Args:
            rag_results: RAG model results
            ft_results: Fine-tuned model results
            rag_summary: RAG model summary
            ft_summary: Fine-tuned model summary
        """
        # Save detailed results
        results = {"rag": rag_results, "ft": ft_results}
        with open(
            self.output_dir / "evaluation_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary = {"rag": rag_summary, "ft": ft_summary}
        with open(
            self.output_dir / "evaluation_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {self.output_dir}")

    def _create_visualizations(
        self, rag_summary: Dict[str, Any], ft_summary: Dict[str, Any]
    ):
        """
        Create visualizations for evaluation results.

        Args:
            rag_summary: RAG model summary
            ft_summary: Fine-tuned model summary
        """
        # Set up plot style
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy comparison
        systems = ["RAG", "Fine-Tuned"]
        accuracies = [rag_summary["accuracy"], ft_summary["accuracy"]]

        ax1.bar(systems, accuracies, color=["#3498db", "#e74c3c"])
        ax1.set_title("Accuracy Comparison")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)

        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f"{v:.2%}", ha="center")

        # Response time comparison
        times = [rag_summary["avg_response_time"], ft_summary["avg_response_time"]]

        ax2.bar(systems, times, color=["#3498db", "#e74c3c"])
        ax2.set_title("Average Response Time")
        ax2.set_ylabel("Time (seconds)")

        for i, v in enumerate(times):
            ax2.text(i, v + 0.01, f"{v:.3f}s", ha="center")

        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_charts.png", dpi=300)

        # Create additional charts if we have question type data
        if "high_confidence_accuracy" in rag_summary:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Accuracy by question type
            question_types = ["High Confidence", "Low Confidence", "Irrelevant"]
            rag_accuracies = [
                rag_summary.get("high_confidence_accuracy", 0),
                rag_summary.get("low_confidence_accuracy", 0),
                rag_summary.get("irrelevant_accuracy", 0),
            ]
            ft_accuracies = [
                ft_summary.get("high_confidence_accuracy", 0),
                ft_summary.get("low_confidence_accuracy", 0),
                ft_summary.get("irrelevant_accuracy", 0),
            ]

            x = range(len(question_types))
            width = 0.35

            ax1.bar(
                [i - width / 2 for i in x],
                rag_accuracies,
                width,
                label="RAG",
                color="#3498db",
            )
            ax1.bar(
                [i + width / 2 for i in x],
                ft_accuracies,
                width,
                label="Fine-Tuned",
                color="#e74c3c",
            )

            ax1.set_title("Accuracy by Question Type")
            ax1.set_ylabel("Accuracy")
            ax1.set_xticks(x)
            ax1.set_xticklabels(question_types)
            ax1.legend()
            ax1.set_ylim(0, 1)

            # Response time by question type
            rag_times = [
                rag_summary.get("high_confidence_avg_time", 0),
                rag_summary.get("low_confidence_avg_time", 0),
                rag_summary.get("irrelevant_avg_time", 0),
            ]
            ft_times = [
                ft_summary.get("high_confidence_avg_time", 0),
                ft_summary.get("low_confidence_avg_time", 0),
                ft_summary.get("irrelevant_avg_time", 0),
            ]

            ax2.bar(
                [i - width / 2 for i in x],
                rag_times,
                width,
                label="RAG",
                color="#3498db",
            )
            ax2.bar(
                [i + width / 2 for i in x],
                ft_times,
                width,
                label="Fine-Tuned",
                color="#e74c3c",
            )

            ax2.set_title("Response Time by Question Type")
            ax2.set_ylabel("Time (seconds)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(question_types)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / "evaluation_charts_by_type.png", dpi=300)
