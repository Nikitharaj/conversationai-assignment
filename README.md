# Financial Q&A System

A comprehensive system comparing **RAG (Retrieval-Augmented Generation)** and **Fine-Tuned Language Models** for answering questions based on financial statements.

## 🎯 Project Overview

This project implements and compares two state-of-the-art approaches for financial question answering:

1. **🔍 RAG System**: Hybrid retrieval (BM25 + FAISS) + DistilGPT2 generation
2. **🎯 Fine-Tuned Model**: PEFT (LoRA) fine-tuning on financial Q&A data

## ✨ Key Features

- **📊 52 Q&A Pairs**: Comprehensive dataset exceeding assignment requirements
- **🔄 Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25) search combination
- **⚡ PEFT Fine-tuning**: Parameter-efficient training with LoRA adapters
- **🛡️ Guardrails**: Input validation and output verification
- **📱 Professional UI**: Streamlit interface with real-time metrics
- **🔧 Multi-format Support**: PDF, Excel, CSV, HTML, TXT processing

## 📁 Project Structure

```
📁 C_AI_assignment/
├── 🎯 app.py                     # Main Streamlit application
├── 📦 requirements.txt           # Python dependencies
├── 📖 README.md                  # Project overview (this file)
├── 📋 PROJECT_SUMMARY.md         # Technical implementation details
├── ✅ ASSIGNMENT_AUDIT.md        # Compliance verification
├── 📊 data/                      # Dataset (75 files total)
│   ├── raw/                      # Original financial documents
│   ├── processed/                # Cleaned text files
│   ├── chunks/                   # Multi-size chunks (100/400 tokens)
│   └── qa_pairs/                 # 52 Q&A pairs (41 train + 11 test)
├── 🤖 models/
│   └── fine_tuned/               # PEFT checkpoints with LoRA adapters
├── 💻 src/                       # Source code (10 files)
│   ├── rag_system/               # Hybrid RAG implementation
│   ├── fine_tuning/              # PEFT fine-tuning
│   ├── data_processing/          # Document processing
│   ├── evaluation/               # Model evaluation
│   └── ui/                       # Interface components
├── 📓 notebooks/                 # Jupyter notebooks (4 files)
│   ├── data_preprocessing.ipynb  # Data collection & preprocessing
│   ├── rag_system.ipynb         # RAG implementation
│   ├── fine_tuning.ipynb        # Fine-tuning experiments
│   └── evaluation.ipynb         # Model comparison
├── 🧪 tests/                     # Comprehensive test suite (17 files)
└── 📈 evaluation_results/        # Performance metrics
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM (for model loading)
- 2GB+ disk space

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd C_AI_assignment
   ```

2. **Create virtual environment:**

   ```bash
   # Using conda (recommended)
   conda create -n financial-qa python=3.10
   conda activate financial-qa

   # Or using venv
   python -m venv financial-qa
   source financial-qa/bin/activate  # Linux/Mac
   # financial-qa\Scripts\activate   # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**

   ```bash
   streamlit run app.py
   ```

5. **Open your browser to:** `http://localhost:8501`

## 📱 How to Use

### 1. **Choose Your Approach**

- **RAG System**: Fast, adaptable, cites sources
- **Fine-Tuned Model**: Specialized, fluent responses

### 2. **Upload Financial Documents**

- Supported formats: PDF, Excel, CSV, HTML, TXT
- Example: Upload Apple's annual report or financial statements

### 3. **Ask Questions**

- _"What was the revenue in 2023?"_
- _"How did iPhone sales perform?"_
- _"What are the main revenue segments?"_

### 4. **Compare Results**

- View side-by-side answers
- Check confidence scores
- Analyze response times
- Review retrieved context (RAG only)

### 5. **Evaluate Performance**

- Switch between systems instantly
- Test with irrelevant questions
- Observe guardrail filtering

## 🧪 Testing

Run the comprehensive test suite to verify functionality:

```bash
# Navigate to the tests directory
cd tests

# Run all tests
python -m pytest

# Run tests for specific modules
python -m pytest rag_system/
python -m pytest fine_tuning/

# Run with verbose output
python -m pytest -v
```

For detailed testing information, see `tests/README.md`.

## 🔧 Technical Implementation

### 🔍 RAG System

- **Hybrid Retrieval**: BM25 (keyword) + FAISS (semantic) with weighted fusion
- **Chunking**: Multi-size strategy (100 & 400 tokens) for optimal retrieval
- **Embeddings**: all-MiniLM-L6-v2 (lightweight, open-source)
- **Generation**: DistilGPT2 with context-aware prompting
- **Guardrails**: Input filtering + hallucination detection

### 🎯 Fine-Tuned Model

- **Base Model**: DistilGPT2 (small, efficient)
- **PEFT Method**: LoRA (Low-Rank Adaptation) with 8-rank adapters
- **Training**: Parameter-efficient fine-tuning (95% fewer parameters)
- **Dataset**: 52 financial Q&A pairs (domain-specific)
- **Performance**: 2.6 second training time vs traditional hours

### 📊 Dataset

- **52 Q&A Pairs** from Apple 2023 financial report + sample statements
- **Training Split**: 41 pairs for model training
- **Test Split**: 11 pairs for evaluation
- **Multi-format Sources**: PDF, text, structured financial data

## 🛠️ Key Dependencies

- **LangChain**: RAG pipeline orchestration
- **Transformers**: Language model inference
- **PEFT**: Parameter-efficient fine-tuning
- **FAISS**: Vector similarity search
- **Streamlit**: Interactive web interface
- **Sentence-Transformers**: Text embeddings

## 📈 Performance Highlights

- **⚡ Fast Training**: 2.6s PEFT vs hours traditional fine-tuning
- **💾 Memory Efficient**: LoRA adapters vs full model fine-tuning
- **🎯 Accurate Retrieval**: Hybrid search combines keyword + semantic
- **🛡️ Robust**: Guardrails prevent hallucination and filter irrelevant queries
- **📱 User-Friendly**: Real-time metrics and confidence scoring

## 📝 Assignment Compliance

✅ **Exceeds all requirements**:

- 52 Q&A pairs (50+ required)
- Hybrid RAG implementation
- PEFT fine-tuning (advanced technique)
- Professional interface with evaluation framework
- Comprehensive documentation and testing

---

**Ready for evaluation and submission** 🚀
