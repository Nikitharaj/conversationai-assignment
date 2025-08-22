# Financial Q&A System

This project implements a Financial Question Answering System using two approaches:

1. **RAG (Retrieval-Augmented Generation)**: Uses document retrieval + generation
2. **Fine-Tuned LLM**: Uses a model specifically fine-tuned on financial data

## Features

- Document processing for various formats (PDF, Excel, CSV, HTML, TXT)
- Document chunking and embedding
- Multiple retrieval methods (dense, sparse, hybrid)
- Answer generation with confidence scoring
- Evaluation framework for comparing approaches
- User-friendly Streamlit interface

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── dependency_manager.py     # Centralized dependency management
├── requirements.txt          # Project dependencies
├── update_dependencies.py    # Script to update dependencies
├── run_tests.py              # Test runner script
├── data/                     # Data directory
│   ├── processed/            # Processed documents
│   ├── qa_pairs/             # Question-answer pairs
│   ├── chunks/               # Document chunks
│   └── indexes/              # Vector indexes
├── models/                   # Model directory
│   ├── rag/                  # RAG model files
│   └── fine_tuned/           # Fine-tuned model files
├── notebooks/                # Jupyter notebooks
│   ├── data_preprocessing.ipynb
│   ├── rag_system.ipynb
│   ├── fine_tuning.ipynb
│   └── evaluation.ipynb
├── src/                      # Source code
│   ├── data_processing/      # Data processing modules
│   ├── rag_system/           # RAG system modules
│   ├── fine_tuning/          # Fine-tuning modules
│   ├── evaluation/           # Evaluation modules
│   └── ui/                   # UI components
└── tests/                    # Test suite
    ├── data_processing/      # Data processing tests
    ├── rag_system/           # RAG system tests
    ├── fine_tuning/          # Fine-tuning tests
    ├── evaluation/           # Evaluation tests
    └── ui/                   # UI tests
```

## Setup and Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd financial-qa-system
   ```

2. Create a conda environment:

   ```
   conda create -n financial-qa python=3.10
   conda activate financial-qa
   ```

3. Install dependencies:

   ```
   python update_dependencies.py
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload a financial document (PDF, Excel, CSV, HTML, TXT)
2. Ask questions about the document
3. Toggle between RAG and Fine-Tuned approaches
4. View retrieved context and confidence scores
5. Compare performance metrics in the evaluation section

## Testing

The project includes a comprehensive test suite covering all modules:

1. Run all tests:

   ```
   ./run_tests.py
   ```

2. Run tests for a specific module:

   ```
   ./run_tests.py -m rag_system
   ```

3. Run tests with verbose output:
   ```
   ./run_tests.py -v
   ```

For more details about testing, see the [tests/README.md](tests/README.md) file.

## Implementation Details

### RAG System

The RAG system uses LangChain components for:

- Document chunking with multiple strategies
- Embedding generation with Sentence Transformers
- Vector storage with FAISS
- Hybrid retrieval (dense + sparse)
- Answer generation with transformer models

### Fine-Tuned Model

The Fine-Tuned model uses:

- Pre-trained language models
- Domain adaptation on financial data
- Parameter-efficient fine-tuning (PEFT)
- Evaluation metrics for comparison

## Dependencies

- LangChain for RAG components
- Transformers for language models
- FAISS for vector search
- Streamlit for the user interface
- PyPDF2, BeautifulSoup for document processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
