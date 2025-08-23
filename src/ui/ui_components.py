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

            # System selection
            system_option = st.radio(
                "Select Q&A System",
                ["RAG", "MoE (Mixture of Experts)"],
                index=0
                if st.session_state.get("current_system", "RAG") == "RAG"
                else 1,
                help="Choose between RAG (retrieval-based) or MoE (specialized expert routing) approach",
            )
            st.session_state.current_system = system_option

            # RAG-specific settings
            if system_option == "RAG":
                st.markdown("**RAG Settings:**")
                chunk_size = st.selectbox(
                    "Chunk Size",
                    [100, 400],
                    index=1,
                    help="Size of document chunks for retrieval",
                )
                st.session_state.chunk_size = chunk_size

                top_k = st.slider(
                    "Top-K Retrieval",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of chunks to retrieve",
                )
                st.session_state.top_k = top_k

            st.markdown("---")

            # Data info
            st.markdown("** Data Info:**")
            st.write("• Company: Sample Financial Reports")
            st.write("• Years: 2022-2023")
            st.write("• Sections: Income, Balance, Cash Flow")

            st.markdown("---")

            # MoE section
            if system_option == "MoE (Mixture of Experts)":
                st.markdown("**MoE Training Options:**")

                # Check if model is available
                if "ft_model" in st.session_state and st.session_state.ft_model:
                    # Get model status
                    try:
                        status = st.session_state.ft_model.get_model_status()

                        # Show current model status
                        if status["is_fine_tuned"]:
                            st.success(f" {status['model_type']}")
                        else:
                            st.info("Base model loaded")

                        # Show available checkpoints
                        if status["available_checkpoints"]:
                            st.write(
                                f" {len(status['available_checkpoints'])} checkpoint(s)"
                            )

                        # Fine-tuning controls
                        st.markdown("**Training:**")

                        # Select training data
                        training_data_options = [
                            "financial_qa_train.json",
                            "comprehensive_qa_pairs.json",
                            "apple_Report_processed_qa_pairs.json",
                            "test_qa_pairs.json",
                        ]

                        selected_data = st.selectbox(
                            "Training Data",
                            training_data_options,
                            help="Choose QA pairs for fine-tuning",
                        )

                        # Training parameters
                        epochs = st.slider(
                            "Epochs", 1, 5, 2, help="Number of training epochs"
                        )
                        batch_size = st.slider(
                            "Batch Size", 1, 8, 2, help="Training batch size"
                        )

                        # Fine-tune button
                        if st.button(
                            " Start Fine-Tuning",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.start_finetuning = True
                            st.session_state.selected_training_data = selected_data
                            st.session_state.training_epochs = epochs
                            st.session_state.training_batch_size = batch_size
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error getting model status: {e}")
                else:
                    st.warning("Fine-tuned model not loaded")

            st.markdown("---")

            # Advanced techniques info
            st.markdown("** Group 118 Techniques:**")
            if system_option == "RAG":
                st.write(" Cross-Encoder Re-ranking")
                st.write("• Hybrid retrieval (dense + sparse)")
                st.write("• Advanced re-ranking")
            else:
                st.write(" Mixture-of-Experts (MoE)")
                st.write("• Specialized expert routing")
                st.write("• PEFT fine-tuning")

                # Show MoE status if available
                if st.session_state.get("ft_model") and hasattr(
                    st.session_state.ft_model, "moe_system"
                ):
                    moe_system = st.session_state.ft_model.moe_system
                    if moe_system:
                        # Debug information
                        st.write(f"Debug: MoE initialized: {moe_system.is_initialized}")
                        st.write(f"Debug: MoE trained: {moe_system.is_trained}")

                        # Check multiple indicators of training completion
                        is_trained = (
                            moe_system.is_trained
                            or st.session_state.get("moe_trained", False)
                            or st.session_state.get("moe_training_completed", False)
                            or
                            # Check if training was successful based on status
                            (
                                hasattr(st.session_state.ft_model, "get_model_status")
                                and st.session_state.ft_model.get_model_status().get(
                                    "model_type"
                                )
                                == "MoE (Trained)"
                            )
                        )

                        if is_trained:
                            st.success("MoE System: **TRAINED**")
                            st.write("• 4 Financial Experts Active")
                            st.write(
                                "• Income Statement, Balance Sheet, Cash Flow, Notes/MD&A"
                            )
                            if hasattr(moe_system, "experts") and moe_system.experts:
                                st.write(f"• {len(moe_system.experts)} experts loaded")
                        else:
                            st.warning("MoE System: **NOT TRAINED**")
                            st.write("• Use Fine-Tuning to train experts")
                    else:
                        st.info("MoE System: **INITIALIZING**")
                else:
                    if not st.session_state.get("ft_model"):
                        st.write("Debug: No ft_model in session_state")
                    elif not hasattr(st.session_state.ft_model, "moe_system"):
                        st.write("Debug: ft_model has no moe_system attribute")

            return None

    @staticmethod
    def render_query_section():
        """Render the query input section."""
        # This method is now handled in the main app
        # Keeping for backward compatibility
        return None

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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            system_name = st.session_state.current_system
            if result.get("model_type"):
                system_name += f" ({result['model_type']})"
            st.metric("System", system_name)
        with col2:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        with col3:
            st.metric("Response Time", f"{result['response_time']:.3f}s")
        with col4:
            method = result.get("method", "Standard")
            st.metric("Method", method)

        # Test Question Analysis (if this was a test question)
        if (
            hasattr(st.session_state, "expected_type")
            and st.session_state.expected_type
        ):
            expected_type = st.session_state.expected_type
            confidence = result["confidence"]

            # Analyze if confidence matches expectations
            confidence_analysis = UIComponents._analyze_test_confidence(
                expected_type, confidence
            )

            st.markdown("---")
            st.markdown("###  Test Question Analysis")

            col_exp, col_actual, col_analysis = st.columns(3)
            with col_exp:
                st.write(f"**Expected:** {expected_type.replace('_', ' ').title()}")
            with col_actual:
                st.write(f"**Actual Confidence:** {confidence:.2%}")
            with col_analysis:
                if confidence_analysis["matches"]:
                    st.success(f" {confidence_analysis['message']}")
                else:
                    st.warning(f" {confidence_analysis['message']}")

            # Clear the expected type after analysis
            st.session_state.expected_type = None

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

    @staticmethod
    def _analyze_test_confidence(
        expected_type: str, actual_confidence: float
    ) -> Dict[str, Any]:
        """
        Analyze if the actual confidence matches the expected confidence level.

        Args:
            expected_type: Expected confidence level (high_confidence, medium_confidence, low_confidence)
            actual_confidence: Actual confidence score (0.0 to 1.0)

        Returns:
            Dictionary with analysis results
        """
        # Define confidence thresholds
        thresholds = {
            "high_confidence": {"min": 0.7, "max": 1.0, "ideal": "above 70%"},
            "medium_confidence": {"min": 0.4, "max": 0.8, "ideal": "40-80%"},
            "low_confidence": {"min": 0.0, "max": 0.5, "ideal": "below 50%"},
        }

        if expected_type not in thresholds:
            return {"matches": False, "message": "Unknown expected type"}

        threshold = thresholds[expected_type]
        min_conf = threshold["min"]
        max_conf = threshold["max"]
        ideal_range = threshold["ideal"]

        matches = min_conf <= actual_confidence <= max_conf

        if matches:
            message = f"Confidence {actual_confidence:.1%} matches expected range ({ideal_range})"
        else:
            if actual_confidence > max_conf:
                message = f"Confidence {actual_confidence:.1%} is higher than expected ({ideal_range})"
            else:
                message = f"Confidence {actual_confidence:.1%} is lower than expected ({ideal_range})"

        return {
            "matches": matches,
            "message": message,
            "expected_range": ideal_range,
            "actual": f"{actual_confidence:.1%}",
        }
