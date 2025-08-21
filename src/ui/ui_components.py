import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class UIComponents:
    """Class containing UI components for the Streamlit application."""

    @staticmethod
    def render_sidebar():
        """Render the sidebar components."""
        with st.sidebar:
            st.header("Settings")

            # File uploader
            uploaded_file = st.file_uploader(
                "Upload Financial Document", type=["pdf", "xlsx", "csv", "html"]
            )
            if uploaded_file:
                st.success(f"Uploaded: {uploaded_file.name}")
                # Return the uploaded file for processing
                return uploaded_file

            # System selection
            system_option = st.radio(
                "Select Q&A System",
                ["RAG", "Fine-Tuned"],
                index=0
                if st.session_state.get("current_system", "RAG") == "RAG"
                else 1,
            )
            st.session_state.current_system = system_option

            # Show evaluation toggle
            show_evaluation = st.toggle(
                "Show Evaluation Results",
                st.session_state.get("show_evaluation", False),
            )
            st.session_state.show_evaluation = show_evaluation

            st.markdown("---")
            st.markdown("### About")
            st.markdown(
                "This application compares RAG and Fine-Tuned approaches for financial Q&A."
            )

            return None

    @staticmethod
    def render_query_section():
        """Render the query input section."""
        query = st.text_input(
            "Ask a question about the financial data:",
            placeholder="e.g., What was the revenue in Q2 2023?",
        )

        return query

    @staticmethod
    def render_answer(result: Dict[str, Any]):
        """
        Render the answer and metadata.

        Args:
            result: Dictionary containing answer and metadata
        """
        st.markdown("### Answer:")
        st.markdown(result["answer"])

        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System", st.session_state.current_system)
        with col2:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        with col3:
            st.metric("Response Time", f"{result['response_time']:.3f}s")

        # Show context (RAG only)
        if st.session_state.current_system == "RAG" and "retrieved_chunks" in result:
            show_context = st.toggle(
                "Show Retrieved Context", st.session_state.get("show_context", False)
            )
            st.session_state.show_context = show_context

            if show_context:
                st.markdown("### Retrieved Context:")

                for i, chunk in enumerate(result["retrieved_chunks"]):
                    with st.expander(
                        f"Chunk {i + 1} (Score: {chunk['score']:.4f}, Method: {chunk['method']})"
                    ):
                        st.markdown(chunk["chunk"]["text"])

    @staticmethod
    def render_evaluation_results(evaluation_dir: Union[str, Path]):
        """
        Render the evaluation results section.

        Args:
            evaluation_dir: Directory containing evaluation results
        """
        evaluation_dir = Path(evaluation_dir)

        # Check if evaluation results exist
        summary_path = evaluation_dir / "evaluation_summary.json"
        if not summary_path.exists():
            st.warning("No evaluation results available.")
            return

        # Load evaluation summary
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        st.markdown("---")
        st.markdown("## Evaluation Results")

        # Display summary table
        st.markdown("### Summary")

        data = {
            "Metric": ["Accuracy", "Avg Response Time (s)", "Avg Confidence"],
            "RAG": [
                f"{summary['rag']['accuracy']:.2%}",
                f"{summary['rag']['avg_response_time']:.3f}s",
                f"{summary['rag']['avg_confidence']:.2%}",
            ],
            "Fine-Tuned": [
                f"{summary['ft']['accuracy']:.2%}",
                f"{summary['ft']['avg_response_time']:.3f}s",
                f"{summary['ft']['avg_confidence']:.2%}",
            ],
        }

        df = pd.DataFrame(data)
        st.table(df)

        # Display charts
        st.markdown("### Visualization")

        # Check if chart image exists
        chart_path = evaluation_dir / "evaluation_charts.png"
        if chart_path.exists():
            st.image(str(chart_path))
        else:
            # Create charts on the fly if image doesn't exist
            UIComponents._create_evaluation_charts(summary)

    @staticmethod
    def _create_evaluation_charts(summary: Dict[str, Any]):
        """
        Create evaluation charts on the fly.

        Args:
            summary: Evaluation summary dictionary
        """
        # Set up plot style
        sns.set(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy comparison
        systems = ["RAG", "Fine-Tuned"]
        accuracies = [summary["rag"]["accuracy"], summary["ft"]["accuracy"]]

        ax1.bar(systems, accuracies, color=["#3498db", "#e74c3c"])
        ax1.set_title("Accuracy Comparison")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)

        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f"{v:.2%}", ha="center")

        # Response time comparison
        times = [
            summary["rag"]["avg_response_time"],
            summary["ft"]["avg_response_time"],
        ]

        ax2.bar(systems, times, color=["#3498db", "#e74c3c"])
        ax2.set_title("Average Response Time")
        ax2.set_ylabel("Time (seconds)")

        for i, v in enumerate(times):
            ax2.text(i, v + 0.01, f"{v:.3f}s", ha="center")

        plt.tight_layout()
        st.pyplot(fig)

        # If we have question type data, show additional charts
        if "high_confidence_accuracy" in summary["rag"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Accuracy by question type
            question_types = ["High Confidence", "Low Confidence", "Irrelevant"]
            rag_accuracies = [
                summary["rag"].get("high_confidence_accuracy", 0),
                summary["rag"].get("low_confidence_accuracy", 0),
                summary["rag"].get("irrelevant_accuracy", 0),
            ]
            ft_accuracies = [
                summary["ft"].get("high_confidence_accuracy", 0),
                summary["ft"].get("low_confidence_accuracy", 0),
                summary["ft"].get("irrelevant_accuracy", 0),
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
                summary["rag"].get("high_confidence_avg_time", 0),
                summary["rag"].get("low_confidence_avg_time", 0),
                summary["rag"].get("irrelevant_avg_time", 0),
            ]
            ft_times = [
                summary["ft"].get("high_confidence_avg_time", 0),
                summary["ft"].get("low_confidence_avg_time", 0),
                summary["ft"].get("irrelevant_avg_time", 0),
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
            st.pyplot(fig)

    @staticmethod
    def render_detailed_results(evaluation_dir: Union[str, Path]):
        """
        Render detailed evaluation results.

        Args:
            evaluation_dir: Directory containing evaluation results
        """
        evaluation_dir = Path(evaluation_dir)

        # Check if detailed results exist
        results_path = evaluation_dir / "evaluation_results.json"
        if not results_path.exists():
            return

        # Load detailed results
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        st.markdown("### Detailed Results")

        # Create tabs for RAG and Fine-Tuned results
        rag_tab, ft_tab = st.tabs(["RAG Results", "Fine-Tuned Results"])

        with rag_tab:
            UIComponents._render_result_table(results["rag"])

        with ft_tab:
            UIComponents._render_result_table(results["ft"])

    @staticmethod
    def _render_result_table(results: List[Dict[str, Any]]):
        """
        Render a table of evaluation results.

        Args:
            results: List of result dictionaries
        """
        # Create a DataFrame
        df = pd.DataFrame(
            [
                {
                    "Question": r["question"],
                    "Answer": r["answer"],
                    "Ground Truth": r["ground_truth"],
                    "Correct": "✓" if r["is_correct"] else "✗",
                    "Confidence": f"{r['confidence']:.2%}",
                    "Response Time": f"{r['response_time']:.3f}s",
                    "Question Type": r.get("question_type", "N/A"),
                }
                for r in results
            ]
        )

        st.dataframe(df, use_container_width=True)
