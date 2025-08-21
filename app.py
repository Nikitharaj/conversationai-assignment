import streamlit as st
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

# Import project modules
from src.data_processing.document_processor import DocumentProcessor
from src.rag_system.rag_system import RAGSystem
from src.fine_tuning.ft_model import FineTunedModel
from src.ui.ui_components import UIComponents

# Set page configuration
st.set_page_config(
    page_title="Financial Q&A Systems",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
QA_PAIRS_DIR = DATA_DIR / "qa_pairs"
RAG_MODEL_DIR = Path("models") / "rag"
FT_MODEL_DIR = Path("models") / "fine_tuned"
EVALUATION_DIR = Path("evaluation_results")

# Create directories if they don't exist
for dir_path in [
    PROCESSED_DIR,
    QA_PAIRS_DIR,
    RAG_MODEL_DIR,
    FT_MODEL_DIR,
    EVALUATION_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize session state
if "current_system" not in st.session_state:
    st.session_state.current_system = "RAG"
if "show_evaluation" not in st.session_state:
    st.session_state.show_evaluation = False
if "show_context" not in st.session_state:
    st.session_state.show_context = False
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "ft_model" not in st.session_state:
    st.session_state.ft_model = None

# Title and description
st.title("Financial Q&A Systems")
st.markdown("""
This application compares two approaches to financial question answering:
- **RAG (Retrieval-Augmented Generation)**: Uses document retrieval + generation
- **Fine-Tuned LLM**: Uses a model specifically fine-tuned on financial data
""")


# Load models
@st.cache_resource
def load_rag_system():
    if RAG_MODEL_DIR.exists() and (RAG_MODEL_DIR / "config.json").exists():
        try:
            rag_system = RAGSystem()
            rag_system.load(RAG_MODEL_DIR)
            return rag_system
        except Exception as e:
            st.error(f"Error loading RAG system: {e}")
            return None
    else:
        st.warning("RAG model not found. Please train the model first.")
        return None


@st.cache_resource
def load_ft_model():
    if FT_MODEL_DIR.exists():
        try:
            ft_model = FineTunedModel(model_path=FT_MODEL_DIR)
            return ft_model
        except Exception as e:
            st.error(f"Error loading Fine-Tuned model: {e}")
            return None
    else:
        st.warning("Fine-Tuned model not found. Please train the model first.")
        return None


# Sidebar
uploaded_file = UIComponents.render_sidebar()

# Load models based on selection
if st.session_state.current_system == "RAG" and st.session_state.rag_system is None:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag_system = load_rag_system()

if (
    st.session_state.current_system == "Fine-Tuned"
    and st.session_state.ft_model is None
):
    with st.spinner("Loading Fine-Tuned model..."):
        st.session_state.ft_model = load_ft_model()

# Process uploaded file
if uploaded_file:
    with st.spinner("Processing uploaded document..."):
        # Save the uploaded file
        temp_file_path = Path("data/raw") / uploaded_file.name
        Path("data/raw").mkdir(parents=True, exist_ok=True)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the document
        processor = DocumentProcessor(output_dir=PROCESSED_DIR)
        try:
            processed_text = processor.process_document(temp_file_path)
            st.success(f"Document processed: {len(processed_text)} characters")

            # If RAG system is loaded, initialize with the new document
            if st.session_state.rag_system:
                # Create chunks
                from src.rag_system.document_chunker import DocumentChunker

                chunker = DocumentChunker(chunk_sizes=[100, 400], chunk_overlap=50)
                chunks_by_size = chunker.chunk_document(processed_text, 400)

                # Add document metadata
                for chunk in chunks_by_size:
                    chunk["document"] = uploaded_file.name

                # Initialize RAG system with chunks
                st.session_state.rag_system.embedding_manager.build_indexes(
                    chunks_by_size
                )
                st.success("RAG system updated with new document")
        except Exception as e:
            st.error(f"Error processing document: {e}")

# Main panel
query = UIComponents.render_query_section()

if query:
    # Input validation
    if len(query) < 5:
        st.warning("Please enter a more specific question.")
    else:
        with st.spinner(f"Processing with {st.session_state.current_system} system..."):
            if st.session_state.current_system == "RAG" and st.session_state.rag_system:
                # Process with RAG system
                result = st.session_state.rag_system.process_query(query)
                UIComponents.render_answer(result)

            elif (
                st.session_state.current_system == "Fine-Tuned"
                and st.session_state.ft_model
            ):
                # Process with Fine-Tuned model
                result = st.session_state.ft_model.process_query(query)
                UIComponents.render_answer(result)

            else:
                st.error(
                    f"The {st.session_state.current_system} system is not loaded. Please check the model files."
                )

# Evaluation results
if st.session_state.show_evaluation:
    if (
        EVALUATION_DIR.exists()
        and (EVALUATION_DIR / "evaluation_summary.json").exists()
    ):
        UIComponents.render_evaluation_results(EVALUATION_DIR)
        UIComponents.render_detailed_results(EVALUATION_DIR)
    else:
        st.warning("No evaluation results available. Please run the evaluation first.")
