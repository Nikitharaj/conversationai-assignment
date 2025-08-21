# Financial Q&A Systems Project

This project implements and compares two financial Q&A systems:

1. RAG (Retrieval-Augmented Generation) system
2. Fine-Tuned Language Model (LLM)

## Project Structure

```
.
├── README.md                  # Project overview
├── requirements.txt           # Python dependencies
├── project_requirements.md    # Detailed project requirements and tracking
├── data/                      # Financial statements and processed data
│   ├── raw/                   # Original financial statements
│   ├── processed/             # Preprocessed text data
│   ├── qa_pairs/              # Generated Q&A pairs
│   ├── chunks/                # Chunked documents for RAG
│   └── indexes/               # Embedding indexes for RAG
├── models/                    # Trained models
│   ├── rag/                   # RAG system model and configuration
│   └── fine_tuned/            # Fine-tuned LLM
├── notebooks/                 # Jupyter notebooks for development
│   ├── data_preprocessing.ipynb
│   ├── rag_system.ipynb
│   ├── fine_tuning.ipynb
│   └── evaluation.ipynb
├── src/                       # Source code
│   ├── data_processing/       # Data processing modules
│   ├── rag_system/            # RAG system implementation
│   ├── fine_tuning/           # Fine-tuning implementation
│   ├── evaluation/            # Evaluation metrics and tools
│   └── ui/                    # Streamlit UI components
├── evaluation_results/        # Evaluation metrics and visualizations
└── app.py                     # Main Streamlit application
```

## Setup and Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd C_AI_assignment
```

2. Create and activate a virtual environment (recommended):

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Data Preprocessing

To preprocess financial documents and generate Q&A pairs:

1. Place your financial statements (PDF, Excel, HTML, CSV) in the `data/raw/` directory
2. Run the data preprocessing notebook:

```bash
jupyter notebook notebooks/data_preprocessing.ipynb
```

3. Execute all cells in the notebook to:
   - Convert documents to plain text
   - Remove noise (headers, footers)
   - Segment by section
   - Generate Q&A pairs
   - Split into training and testing sets

### 2. RAG System

To build and train the RAG system:

```bash
jupyter notebook notebooks/rag_system.ipynb
```

Execute all cells to:

- Chunk documents into smaller segments
- Create embeddings and indexes
- Test different retrieval methods
- Set up answer generation
- Save the complete RAG system

### 3. Fine-Tuned Model

To fine-tune the language model:

```bash
jupyter notebook notebooks/fine_tuning.ipynb
```

Execute all cells to:

- Load and analyze Q&A pairs
- Initialize the fine-tuner
- Evaluate the base model
- Fine-tune the model
- Evaluate the fine-tuned model
- Compare pre-training and post-training results

### 4. Evaluation

To evaluate and compare both systems:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

Execute all cells to:

- Load both models
- Evaluate on test set
- Visualize evaluation results
- Evaluate on official test questions

### 5. Streamlit Application

To run the complete application with UI:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Using the Streamlit Application

1. **Upload Documents**:

   - Use the sidebar to upload financial documents (PDF, Excel, HTML, CSV)
   - The system will process the document and make it available for querying

2. **Select Q&A System**:

   - Choose between RAG and Fine-Tuned systems using the radio buttons in the sidebar

3. **Ask Questions**:

   - Type your financial question in the input field
   - View the answer, confidence score, and response time
   - For RAG system, you can toggle to show the retrieved context

4. **View Evaluation Results**:
   - Toggle "Show Evaluation Results" in the sidebar
   - View accuracy and response time comparisons
   - Explore detailed evaluation metrics

## Running Individual Components

### Data Processing

To process a single document:

```python
from src.data_processing.document_processor import DocumentProcessor

processor = DocumentProcessor(output_dir="data/processed")
processor.process_document("data/raw/your_document.pdf")
```

### RAG System

To use the RAG system directly:

```python
from src.rag_system.rag_system import RAGSystem

# Load the system
rag_system = RAGSystem()
rag_system.load("models/rag")

# Process a query
result = rag_system.process_query("What was the revenue in Q2 2023?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Response time: {result['response_time']:.3f}s")
```

### Fine-Tuned Model

To use the fine-tuned model directly:

```python
from src.fine_tuning.ft_model import FineTunedModel

# Load the model
ft_model = FineTunedModel(model_path="models/fine_tuned")

# Process a query
result = ft_model.process_query("What was the revenue in Q2 2023?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Response time: {result['response_time']:.3f}s")
```

## Evaluation Metrics

The project compares both systems on:

- **Accuracy**: Correctness of answers compared to ground truth
- **Inference Speed**: Response time for generating answers
- **Robustness**: Handling of irrelevant or out-of-domain queries

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FAISS
- Streamlit
- See `requirements.txt` for complete list of dependencies
