"""
Financial Q&A System Application

This Streamlit application provides a user interface for the Financial Q&A System,
which uses Retrieval-Augmented Generation (RAG) and Fine-Tuned LLM approaches
to answer financial questions based on document context.
"""

import streamlit as st
import time
import json
import pandas as pd
from pathlib import Path
import warnings


# Suppress specific warnings
warnings.filterwarnings(
    "ignore", message=".*torch.utils._pytree._register_pytree_node.*"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings(
    "ignore", message=".*bitsandbytes.*compiled without GPU support.*"
)
warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")

# Apply dependency management - DISABLED to avoid version conflicts
# import dependency_manager  # Commented out - using correct versions directly

# Define availability flags at the top level
langchain_available = False
langchain_fine_tuning_available = False

# Import the integrated RAG implementation
try:
    from src.rag_system.integrated_rag import IntegratedRAG

    langchain_available = True
except ImportError as e:
    warnings.warn(
        f"Could not import IntegratedRAG: {e}. Falling back to simplified implementation."
    )
    langchain_available = False

    # Try to import RAG as fallback
    try:
        from src.rag_system.rag import RAG

        langchain_available = True
    except ImportError as e:
        warnings.warn(f"Could not import RAG: {e}. Using fallback.")
        langchain_available = False

# --- Import project modules with fallbacks ---
try:
    from src.rag_system.document_chunker import DocumentChunker
except ImportError as e:
    warnings.warn(f"Could not import DocumentChunker: {e}. Using fallback.")

    # Define a simple document chunker as fallback
    class DocumentChunker:
        def __init__(self, chunk_size=400, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def chunk_document(self, text, chunk_size=None):
            chunk_size = chunk_size or self.chunk_size
            return [{"text": text, "start_idx": 0, "end_idx": len(text)}]


try:
    from src.data_processing.document_processor import DocumentProcessor
except ImportError as e:
    warnings.warn(f"Could not import DocumentProcessor: {e}. Using fallback.")

    class DocumentProcessor:
        def __init__(self, output_dir=None):
            self.output_dir = output_dir

        def process_document(self, file_path):
            return "Document processing not available due to import errors."


# Define a fallback RAG system for when LangChain is not available
class FallbackRAGSystem:
    """Fallback RAG system when LangChain is not available."""

    def __init__(
        self,
        embedding_model=None,
        llm_model=None,
        retrieval_method="sparse",
        top_k=3,
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.retrieval_method = retrieval_method
        self.top_k = top_k
        self.is_initialized = False
        self.embedding_manager = None

        # Create dummy embedding_manager with build_indexes method
        class DummyEmbeddingManager:
            def build_indexes(self, chunks):
                pass

            def search(self, query, top_k=3, search_type="sparse"):
                return []

        self.embedding_manager = DummyEmbeddingManager()

    def process_query(self, query):
        return {
            "query": query,
            "answer": "I'm sorry, the RAG system is not available.",
            "confidence": 0.0,
            "response_time": time.time(),
            "retrieved_chunks": [],
        }


try:
    from src.fine_tuning.fine_tuner import FineTuner

    langchain_fine_tuning_available = True
except ImportError as e:
    warnings.warn(f"Could not import FineTuner: {e}. Using fallback.")
    langchain_fine_tuning_available = False

    try:
        from src.fine_tuning.ft_model import FineTunedModel
    except ImportError as e:
        warnings.warn(f"Could not import FineTunedModel: {e}. Using fallback.")

        class FineTunedModel:
            def __init__(self, model_path=None):
                self.model_path = model_path

            def process_query(self, query):
                return {
                    "query": query,
                    "answer": "I'm sorry, the fine-tuned model is not available.",
                    "confidence": 0.0,
                    "response_time": time.time(),
                }


try:
    from src.ui.ui_components import UIComponents
except ImportError as e:
    warnings.warn(f"Could not import UIComponents: {e}. Using fallback.")

    class UIComponents:
        @staticmethod
        def render_sidebar():
            """Render the sidebar components."""
            with st.sidebar:
                st.header("Settings")

                # File uploader
                uploaded_file = st.file_uploader(
                    "Upload Financial Document",
                    type=["pdf", "xlsx", "csv", "html", "txt"],
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
        def render_answer(result):
            """Render the answer and metadata."""
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
            if (
                st.session_state.current_system == "RAG"
                and "retrieved_chunks" in result
            ):
                show_context = st.toggle(
                    "Show Retrieved Context",
                    st.session_state.get("show_context", False),
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
        def render_evaluation_results(evaluation_dir):
            """Render the evaluation results section."""
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

        @staticmethod
        def render_detailed_results(evaluation_dir):
            """Render detailed evaluation results."""
            st.markdown("### Detailed Results")
            st.info("Detailed results not available in fallback mode.")


# --- Main App Logic ---

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
    DATA_DIR,
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


# Load models using Streamlit's caching
@st.cache_resource
def load_rag_system():
    global langchain_available
    try:
        # Try to use Integrated RAG if available
        if langchain_available:
            st.info("Using Integrated RAG system")
            # Create IntegratedRAG with realistic settings
            rag_system = IntegratedRAG(
                embedding_model="all-MiniLM-L6-v2",  # Smaller embedding model
                llm_model="distilgpt2",  # Use lightweight but real LLM
                retrieval_method="sparse",  # Less memory-intensive retrieval
                top_k=3,  # Retrieve fewer chunks
            )

            # Initialize with empty state - documents will be loaded when uploaded
            st.info("RAG system ready. Please upload a document to begin analysis.")

            # Initialize with minimal sample for system readiness
            sample_chunks = [
                {
                    "text": "Please upload a financial document to begin analysis.",
                    "document": "system_message.txt",
                }
            ]
            rag_system.embedding_manager.build_indexes(sample_chunks)

            rag_system.is_initialized = True
            return rag_system
        else:
            # Try to use RAG as fallback
            try:
                # Already imported at the top as RAG

                # Update the flag if import was successful
                langchain_available = True

                st.info("Using RAG system")
                # Create RAG with realistic settings
                rag_system = RAG(
                    embedding_model_name="all-MiniLM-L6-v2",  # Smaller embedding model
                    llm_model_name="distilgpt2",  # Use lightweight but real LLM
                    retrieval_method="sparse",  # Less memory-intensive retrieval
                    top_k=3,  # Retrieve fewer chunks
                )
                # Initialize with empty chunks
                rag_system.build_indexes([])
                return rag_system
            except ImportError:
                pass

            st.warning(
                "LangChain not available. Using fallback RAG system with limited functionality."
            )
            # Use fallback RAG system with realistic settings
            rag_system = FallbackRAGSystem(
                embedding_model="all-MiniLM-L6-v2",  # Smaller embedding model
                llm_model="distilgpt2",  # Use lightweight but real LLM
                retrieval_method="sparse",  # Less memory-intensive retrieval
                top_k=3,  # Retrieve fewer chunks
            )
            # Mark as initialized to skip the check
            rag_system.is_initialized = True
            return rag_system
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        import traceback

        traceback.print_exc()

        # Create a fallback
        fallback = FallbackRAGSystem(llm_model=None)
        fallback.embedding_manager.build_indexes([])
        fallback.is_initialized = True
        return fallback


@st.cache_resource
def load_ft_model():
    global langchain_fine_tuning_available
    try:
        # Try to use FineTuner if available
        if langchain_fine_tuning_available:
            st.info("Using Fine-Tuned model")
            ft_model = FineTuner()  # Use default "distilgpt2" model
            return ft_model
        else:
            st.warning(
                "Advanced Fine-Tuning not available. Using basic Fine-Tuned model."
            )
            ft_model = FineTunedModel(model_path=FT_MODEL_DIR)
            return ft_model
    except Exception as e:
        st.error(f"Error loading Fine-Tuned model: {e}")
        import traceback

        traceback.print_exc()

        # Return fallback
        try:
            return FineTunedModel()
        except Exception:

            class FallbackModel:
                def process_query(self, query):
                    return {
                        "query": query,
                        "answer": "Fine-tuned model not available.",
                        "confidence": 0.0,
                        "response_time": 0.0,
                    }

            return FallbackModel()


# Sidebar
uploaded_file = UIComponents.render_sidebar()

# Load the selected system only when a document is uploaded
if uploaded_file:
    if st.session_state.current_system == "RAG" and st.session_state.rag_system is None:
        st.session_state.rag_system = load_rag_system()

    if (
        st.session_state.current_system == "Fine-Tuned"
        and st.session_state.ft_model is None
    ):
        st.session_state.ft_model = load_ft_model()

# Document processing section
if uploaded_file:
    try:
        # Save uploaded file
        file_bytes = uploaded_file.read()
        file_path = Path(PROCESSED_DIR) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Process document
        processor = DocumentProcessor(output_dir=PROCESSED_DIR)
        processed_text = processor.process_document(file_path)

        # Display processing result
        st.success(f"Document processed: {len(processed_text)} characters")

        # Handle Fine-Tuning if Fine-Tuned system is selected
        if (
            st.session_state.current_system == "Fine-Tuned"
            and st.session_state.ft_model
        ):
            st.info("🚀 Starting fine-tuning process on uploaded document...")

            try:
                # Generate Q&A pairs from the uploaded document
                with st.spinner("Generating Q&A pairs from document..."):
                    from src.data_processing.qa_generator import QAGenerator

                    qa_generator = QAGenerator(output_dir=Path("data/qa_pairs"))

                    # Save processed text to a temporary file for QA generation
                    # Extract filename without extension from uploaded file
                    file_name_without_ext = Path(uploaded_file.name).stem
                    temp_text_file = (
                        Path(PROCESSED_DIR) / f"{file_name_without_ext}_processed.txt"
                    )
                    with open(temp_text_file, "w", encoding="utf-8") as f:
                        f.write(processed_text)

                    # Generate Q&A pairs (reduced number for faster fine-tuning)
                    qa_pairs = qa_generator.generate_qa_pairs(
                        temp_text_file, num_pairs=20
                    )

                    st.success(f"Generated {len(qa_pairs)} Q&A pairs from document")

                # Save Q&A pairs for fine-tuning
                qa_file = (
                    Path("data/qa_pairs") / f"{file_name_without_ext}_qa_pairs.json"
                )
                qa_file.parent.mkdir(parents=True, exist_ok=True)

                with open(qa_file, "w", encoding="utf-8") as f:
                    json.dump(qa_pairs, f, indent=2)

                # Fine-tune the model
                with st.spinner(
                    "Fine-tuning model on document (this may take a few minutes)..."
                ):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Create a new fine-tuner with faster settings for on-the-fly training
                    from src.fine_tuning.fine_tuner import FineTuner

                    # Try with PEFT first, fall back to without PEFT if needed
                    fast_fine_tuner = FineTuner(
                        model_name="distilgpt2",
                        output_dir=Path("models/fine_tuned") / file_name_without_ext,
                        use_peft=True,
                    )

                    # Check if the fine tuner was initialized properly
                    if (
                        not hasattr(fast_fine_tuner, "model")
                        or fast_fine_tuner.model is None
                    ):
                        st.warning(
                            "PEFT not available, using standard fine-tuning (may take longer)"
                        )
                        fast_fine_tuner = FineTuner(
                            model_name="distilgpt2",
                            output_dir=Path("models/fine_tuned")
                            / file_name_without_ext,
                            use_peft=False,
                        )

                    # Set faster training parameters for on-the-fly training
                    fast_fine_tuner.num_epochs = 1  # Quick training
                    fast_fine_tuner.batch_size = 2  # Small batch for memory efficiency
                    fast_fine_tuner.learning_rate = 1e-4  # Conservative learning rate

                    status_text.text("Initializing model...")
                    progress_bar.progress(0.2)

                    status_text.text("Preparing training data...")
                    progress_bar.progress(0.4)

                    status_text.text("Fine-tuning model...")
                    progress_bar.progress(0.6)

                    # Perform quick fine-tuning directly with Q&A pairs
                    success = fast_fine_tuner.quick_fine_tune(qa_pairs)

                    progress_bar.progress(1.0)
                    status_text.text("Fine-tuning complete!")

                    if success:
                        # Update the session state with the newly fine-tuned model
                        st.session_state.ft_model = fast_fine_tuner

                        st.success("🎉 Model successfully fine-tuned on your document!")
                        st.info(
                            "The model is now specialized for answering questions about your uploaded document."
                        )

                        # Show some example Q&A pairs that were used for training
                        with st.expander("View training Q&A pairs"):
                            for i, pair in enumerate(
                                qa_pairs[:5]
                            ):  # Show first 5 pairs
                                st.write(f"**Q{i + 1}:** {pair['question']}")
                                st.write(f"**A{i + 1}:** {pair['answer']}")
                                st.write("---")
                    else:
                        st.warning(
                            "Fine-tuning encountered issues. Using pre-trained model responses."
                        )
                        # Still use the fast_fine_tuner but without the custom training
                        st.session_state.ft_model = fast_fine_tuner

                    # Clean up temporary file
                    if temp_text_file.exists():
                        temp_text_file.unlink()

            except Exception as e:
                st.error(f"Error during fine-tuning: {e}")
                st.warning("Falling back to pre-trained model responses.")
                import traceback

                traceback.print_exc()

        # Chunk document and update RAG system
        elif st.session_state.rag_system:
            try:
                # Try to use DocumentChunker
                from src.rag_system.document_chunker import DocumentChunker

                chunker = DocumentChunker(chunk_sizes=[400], chunk_overlap=50)
                chunks = chunker.chunk_document(processed_text)

                # Add document metadata
                for chunk in chunks:
                    chunk["document"] = uploaded_file.name
            except Exception as e:
                st.warning(
                    f"Error using DocumentChunker: {e}. Falling back to simple chunker."
                )
                # Simple chunking as fallback
                chunks = [{"text": processed_text, "document": uploaded_file.name}]

            # Handle different RAG system types
            if isinstance(st.session_state.rag_system, IntegratedRAG):
                # LangChain Integrated RAG system
                st.session_state.rag_system.embedding_manager.build_indexes(chunks)
                st.session_state.rag_system.is_initialized = True
            elif hasattr(st.session_state.rag_system, "build_indexes"):
                # LangChain RAG system
                st.session_state.rag_system.build_indexes(chunks)
            else:
                # Original RAG system
                st.session_state.rag_system.embedding_manager.build_indexes(chunks)

            st.success("RAG system updated with new document")
    except Exception as e:
        st.error(f"Error processing document: {e}")
        import traceback

        traceback.print_exc()

# Main panel for Q&A
if uploaded_file:
    # Only show query interface if a document is uploaded
    query = UIComponents.render_query_section()

    if query:
        if len(query) < 5:
            st.warning("Please enter a more specific question.")
        else:
            system_choice = st.session_state.current_system
            with st.spinner(f"Processing with {system_choice} system..."):
                result = None
                if system_choice == "RAG" and st.session_state.rag_system:
                    result = st.session_state.rag_system.process_query(query)
                elif system_choice == "Fine-Tuned" and st.session_state.ft_model:
                    result = st.session_state.ft_model.process_query(query)
                else:
                    st.error(f"The {system_choice} system is not loaded.")

                if result:
                    UIComponents.render_answer(result)
else:
    # Show message to upload document
    st.info(
        "👆 Please upload a financial document using the sidebar to begin analysis."
    )

    # Show different instructions based on selected system
    if st.session_state.current_system == "Fine-Tuned":
        st.markdown("""
        ### 🚀 Fine-Tuned Mode
        When you upload a document in Fine-Tuned mode, the app will:
        1. **Generate Q&A pairs** from your document automatically
        2. **Fine-tune the model** on your specific document (takes 2-3 minutes)
        3. **Provide specialized answers** based on your document's content
        
        This creates a model that's specifically trained on your document!
        """)
    else:
        st.markdown("""
        ### 🔍 RAG Mode
        RAG (Retrieval-Augmented Generation) will:
        1. **Process and chunk** your uploaded document
        2. **Search relevant sections** for each question
        3. **Generate answers** based on retrieved content
        
        This provides fast, accurate responses without model training.
        """)

    st.markdown("""
    ### Supported File Types:
    - **PDF**: Financial reports, earnings statements
    - **Excel/CSV**: Financial data spreadsheets  
    - **HTML**: Web-based financial documents
    
    ### Example Questions:
    - What was the revenue for Q3 2023?
    - How did expenses change compared to last year?
    - What are the key financial highlights?
    """)

# Evaluation results section
if st.session_state.get("show_evaluation"):
    eval_summary_path = EVALUATION_DIR / "evaluation_summary.json"
    if eval_summary_path.exists():
        UIComponents.render_evaluation_results(EVALUATION_DIR)
        UIComponents.render_detailed_results(EVALUATION_DIR)
    else:
        st.warning("No evaluation results available. Please run the evaluation first.")
