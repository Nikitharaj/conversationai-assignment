import os
import json
import time
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..rag_system.rag_system import RAGSystem
from ..fine_tuning.ft_model import FineTunedModel

class Evaluator:
    """Class for evaluating and comparing RAG and Fine-Tuned models."""
    
    def __init__(
        self,
        rag_system: RAGSystem,
        ft_model: FineTunedModel,
        output_dir: Union[str, Path] = "evaluation_results"
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_system: Initialized RAG system
            ft_model: Initialized Fine-Tuned model
            output_dir: Directory to save evaluation results
        """
        self.rag_system = rag_system
        self.ft_model = ft_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation results
        self.results = {
            "rag": [],
            "ft": []
        }
    
    def evaluate_test_set(self, test_file: Union[str, Path]):
        """
        Evaluate both systems on a test set of Q&A pairs.
        
        Args:
            test_file: Path to a JSON file containing test Q&A pairs
        """
        test_file = Path(test_file)
        
        # Load test Q&A pairs
        with open(test_file, 'r', encoding='utf-8') as f:
            test_pairs = json.load(f)
        
        # Evaluate each test pair
        for i, pair in enumerate(test_pairs):
            print(f"Evaluating question {i+1}/{len(test_pairs)}: {pair['question']}")
            
            # Evaluate RAG system
            rag_result = self.rag_system.process_query(pair["question"])
            
            # Evaluate Fine-Tuned model
            ft_result = self.ft_model.process_query(pair["question"])
            
            # Calculate accuracy (simple string similarity)
            from difflib import SequenceMatcher
            
            rag_similarity = SequenceMatcher(None, rag_result["answer"].lower(), pair["answer"].lower()).ratio()
            ft_similarity = SequenceMatcher(None, ft_result["answer"].lower(), pair["answer"].lower()).ratio()
            
            rag_is_correct = rag_similarity > 0.5
            ft_is_correct = ft_similarity > 0.5
            
            # Store results
            self.results["rag"].append({
                "question": pair["question"],
                "ground_truth": pair["answer"],
                "answer": rag_result["answer"],
                "confidence": rag_result["confidence"],
                "response_time": rag_result["response_time"],
                "similarity": rag_similarity,
                "is_correct": rag_is_correct
            })
            
            self.results["ft"].append({
                "question": pair["question"],
                "ground_truth": pair["answer"],
                "answer": ft_result["answer"],
                "confidence": ft_result["confidence"],
                "response_time": ft_result["response_time"],
                "similarity": ft_similarity,
                "is_correct": ft_is_correct
            })
        
        # Save results
        with open(self.output_dir / "evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        self._generate_summary()
    
    def evaluate_official_questions(self, questions: List[Dict[str, str]]):
        """
        Evaluate both systems on the official evaluation questions.
        
        Args:
            questions: List of dictionaries containing questions and ground truth answers
        """
        # Evaluate each question
        for i, q_data in enumerate(questions):
            question = q_data["question"]
            ground_truth = q_data["answer"]
            question_type = q_data["type"]  # "high_confidence", "low_confidence", or "irrelevant"
            
            print(f"Evaluating official question {i+1}/{len(questions)}: {question}")
            
            # Evaluate RAG system
            rag_result = self.rag_system.process_query(question)
            
            # Evaluate Fine-Tuned model
            ft_result = self.ft_model.process_query(question)
            
            # Calculate accuracy (simple string similarity)
            from difflib import SequenceMatcher
            
            rag_similarity = SequenceMatcher(None, rag_result["answer"].lower(), ground_truth.lower()).ratio()
            ft_similarity = SequenceMatcher(None, ft_result["answer"].lower(), ground_truth.lower()).ratio()
            
            rag_is_correct = rag_similarity > 0.5
            ft_is_correct = ft_similarity > 0.5
            
            # Store results
            self.results["rag"].append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": rag_result["answer"],
                "confidence": rag_result["confidence"],
                "response_time": rag_result["response_time"],
                "similarity": rag_similarity,
                "is_correct": rag_is_correct,
                "question_type": question_type
            })
            
            self.results["ft"].append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": ft_result["answer"],
                "confidence": ft_result["confidence"],
                "response_time": ft_result["response_time"],
                "similarity": ft_similarity,
                "is_correct": ft_is_correct,
                "question_type": question_type
            })
        
        # Save results
        with open(self.output_dir / "official_evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate a summary of evaluation results."""
        # Calculate overall metrics
        rag_correct = sum(1 for r in self.results["rag"] if r["is_correct"])
        ft_correct = sum(1 for r in self.results["ft"] if r["is_correct"])
        
        rag_accuracy = rag_correct / len(self.results["rag"]) if self.results["rag"] else 0
        ft_accuracy = ft_correct / len(self.results["ft"]) if self.results["ft"] else 0
        
        rag_avg_time = sum(r["response_time"] for r in self.results["rag"]) / len(self.results["rag"]) if self.results["rag"] else 0
        ft_avg_time = sum(r["response_time"] for r in self.results["ft"]) / len(self.results["ft"]) if self.results["ft"] else 0
        
        rag_avg_confidence = sum(r["confidence"] for r in self.results["rag"]) / len(self.results["rag"]) if self.results["rag"] else 0
        ft_avg_confidence = sum(r["confidence"] for r in self.results["ft"]) / len(self.results["ft"]) if self.results["ft"] else 0
        
        # Create summary dictionary
        summary = {
            "rag": {
                "accuracy": rag_accuracy,
                "avg_response_time": rag_avg_time,
                "avg_confidence": rag_avg_confidence,
                "correct_count": rag_correct,
                "total_count": len(self.results["rag"])
            },
            "ft": {
                "accuracy": ft_accuracy,
                "avg_response_time": ft_avg_time,
                "avg_confidence": ft_avg_confidence,
                "correct_count": ft_correct,
                "total_count": len(self.results["ft"])
            }
        }
        
        # Check if we have question types
        if all("question_type" in r for r in self.results["rag"]):
            # Calculate metrics by question type
            question_types = set(r["question_type"] for r in self.results["rag"])
            
            for q_type in question_types:
                # RAG metrics by question type
                rag_by_type = [r for r in self.results["rag"] if r["question_type"] == q_type]
                rag_correct_by_type = sum(1 for r in rag_by_type if r["is_correct"])
                rag_accuracy_by_type = rag_correct_by_type / len(rag_by_type) if rag_by_type else 0
                rag_avg_time_by_type = sum(r["response_time"] for r in rag_by_type) / len(rag_by_type) if rag_by_type else 0
                
                # FT metrics by question type
                ft_by_type = [r for r in self.results["ft"] if r["question_type"] == q_type]
                ft_correct_by_type = sum(1 for r in ft_by_type if r["is_correct"])
                ft_accuracy_by_type = ft_correct_by_type / len(ft_by_type) if ft_by_type else 0
                ft_avg_time_by_type = sum(r["response_time"] for r in ft_by_type) / len(ft_by_type) if ft_by_type else 0
                
                # Add to summary
                summary["rag"][f"{q_type}_accuracy"] = rag_accuracy_by_type
                summary["rag"][f"{q_type}_avg_time"] = rag_avg_time_by_type
                summary["ft"][f"{q_type}_accuracy"] = ft_accuracy_by_type
                summary["ft"][f"{q_type}_avg_time"] = ft_avg_time_by_type
        
        # Save summary
        with open(self.output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(summary)
    
    def _create_visualizations(self, summary: Dict[str, Any]):
        """
        Create visualizations of evaluation results.
        
        Args:
            summary: Summary dictionary of evaluation results
        """
        # Set up plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        systems = ["RAG", "Fine-Tuned"]
        accuracies = [summary["rag"]["accuracy"], summary["ft"]["accuracy"]]
        plt.bar(systems, accuracies, color=["#3498db", "#e74c3c"])
        plt.title("Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
        
        # Response time comparison
        plt.subplot(2, 2, 2)
        times = [summary["rag"]["avg_response_time"], summary["ft"]["avg_response_time"]]
        plt.bar(systems, times, color=["#3498db", "#e74c3c"])
        plt.title("Average Response Time")
        plt.ylabel("Time (seconds)")
        
        for i, v in enumerate(times):
            plt.text(i, v + 0.01, f"{v:.3f}s", ha='center')
        
        # Check if we have question type data
        if "high_confidence_accuracy" in summary["rag"]:
            # Accuracy by question type
            plt.subplot(2, 2, 3)
            question_types = ["High Confidence", "Low Confidence", "Irrelevant"]
            rag_accuracies = [
                summary["rag"].get("high_confidence_accuracy", 0),
                summary["rag"].get("low_confidence_accuracy", 0),
                summary["rag"].get("irrelevant_accuracy", 0)
            ]
            ft_accuracies = [
                summary["ft"].get("high_confidence_accuracy", 0),
                summary["ft"].get("low_confidence_accuracy", 0),
                summary["ft"].get("irrelevant_accuracy", 0)
            ]
            
            x = np.arange(len(question_types))
            width = 0.35
            
            plt.bar(x - width/2, rag_accuracies, width, label="RAG", color="#3498db")
            plt.bar(x + width/2, ft_accuracies, width, label="Fine-Tuned", color="#e74c3c")
            
            plt.title("Accuracy by Question Type")
            plt.ylabel("Accuracy")
            plt.xticks(x, question_types)
            plt.legend()
            plt.ylim(0, 1)
            
            # Response time by question type
            plt.subplot(2, 2, 4)
            rag_times = [
                summary["rag"].get("high_confidence_avg_time", 0),
                summary["rag"].get("low_confidence_avg_time", 0),
                summary["rag"].get("irrelevant_avg_time", 0)
            ]
            ft_times = [
                summary["ft"].get("high_confidence_avg_time", 0),
                summary["ft"].get("low_confidence_avg_time", 0),
                summary["ft"].get("irrelevant_avg_time", 0)
            ]
            
            plt.bar(x - width/2, rag_times, width, label="RAG", color="#3498db")
            plt.bar(x + width/2, ft_times, width, label="Fine-Tuned", color="#e74c3c")
            
            plt.title("Response Time by Question Type")
            plt.ylabel("Time (seconds)")
            plt.xticks(x, question_types)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_charts.png", dpi=300)
        
        # Create a summary table as CSV
        df = pd.DataFrame({
            "Metric": ["Accuracy", "Avg Response Time (s)", "Avg Confidence"],
            "RAG": [summary["rag"]["accuracy"], summary["rag"]["avg_response_time"], summary["rag"]["avg_confidence"]],
            "Fine-Tuned": [summary["ft"]["accuracy"], summary["ft"]["avg_response_time"], summary["ft"]["avg_confidence"]]
        })
        
        df.to_csv(self.output_dir / "evaluation_summary.csv", index=False)


if __name__ == "__main__":
    # Example usage
    # rag_system = RAGSystem(...)
    # ft_model = FineTunedModel(...)
    # evaluator = Evaluator(rag_system, ft_model, output_dir="../../evaluation_results")
    
    # Evaluate on test set
    # evaluator.evaluate_test_set("../../data/qa_pairs/financial_qa_test.json")
    
    # Evaluate on official questions
    # official_questions = [
    #     {"question": "What was the revenue in Q2 2023?", "answer": "The revenue was $10.5 million.", "type": "high_confidence"},
    #     {"question": "How does the profit margin compare to industry average?", "answer": "The profit margin was 2% below the industry average.", "type": "low_confidence"},
    #     {"question": "What is your favorite color?", "answer": "I can only answer questions related to financial information in the provided documents.", "type": "irrelevant"}
    # ]
    # evaluator.evaluate_official_questions(official_questions)
