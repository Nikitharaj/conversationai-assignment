# Financial Q&A Systems Project Requirements - Implementation Status

## 🎯 Project Objective (Completed)

Build and compare two financial Q&A systems based on company statements:

- RAG (Retrieval-Augmented Generation) chatbot
- Fine-Tuned Language Model (LLM) chatbot

Compare performance on:

- Accuracy
- Inference speed
- Robustness to irrelevant input

## 🧩 Functional Requirements (Implemented)

### 1. Data Collection & Preprocessing

- **Source**: Public or private financial statements (last 2 years)
- **Format**: PDF, Excel, HTML → Convert to plain text
- **Preprocessing**:
  - OCR/parsing
  - Remove noise (headers, footers, page numbers)
  - Segment by section (income statement, balance sheet, etc.)
- **Q&A Pairs**:
  - Construct 50 Q/A pairs derived from the data
  - Use 40 for training (Fine-Tuned model)
  - Use 10 for testing (both systems)

### 2. RAG System

- **Chunking**: Two sizes (e.g., 100 and 400 tokens)
- **Embedding**: Use all-MiniLM-L6-v2 or similar
- **Retrieval**:
  - Dense index (FAISS or Chroma)
  - Sparse index (BM25 or TF-IDF)
  - Combine via hybrid retrieval (score fusion or union)
- **Advanced Technique** (based on Group # mod 5):
  - Options: Chunk merging, re-ranking, memory-augmented, etc.
- **Answer Generation**:
  - Use small LLM (DistilGPT2, GPT-2 Small, etc.)
  - Concatenate retrieved chunks + query
- **Guardrails**:
  - Input-side: Filter irrelevant or unsafe queries
  - Output-side: Detect hallucinations or unsupported claims

### 3. Fine-Tuned Model

- **Format**: 50 Q/A pairs for fine-tuning (40 train / 10 test)
- **Model Selection**: Choose small open-source model:
  - MiniLM, DistilBERT, GPT-2 Small, Falcon 7B, Mistral 7B
- **Pre-training Evaluation**: Use 10 test questions before fine-tuning
- **Fine-tuning Process**:
  - Log: learning rate, batch size, epochs, hardware
  - Use assigned advanced fine-tuning method (Group # mod 5)
- **Guardrails**: Input or output-side, as in RAG

### 4. Evaluation

- **Train/Test Split**:
  - Train FT model on 40 Q/A pairs
  - Use remaining 10 Q/As to test both systems
- **Official Questions**:
  - High-confidence
  - Low-confidence
  - Irrelevant
- **Metrics** (for each question):
  - Ground truth answer
  - Model answer
  - Confidence score
  - Response time
  - Correct (Y/N)

## 💻 UI Requirements (Implemented)

- **Platform**: Streamlit
- **Layout**:
  - **Sidebar**:
    - Upload document
    - Select mode: RAG / Fine-Tuned
    - Toggle: Show Evaluation Results
  - **Main Panel**:
    - Text input: User query
    - Output display:
      - Answer
      - Confidence score
      - Inference time
      - System used (RAG or FT)
    - Toggle: "Show/Hide Retrieved Context" (for RAG mode only)
- **Features**:
  - Query validation
  - Interactive system toggle
  - Optional transparency via chunk display (RAG)
  - Visual output of evaluation summary (charts, tables)

## 📦 Deliverables (Current Status)

| Component           | Description                                                     | Status      |
| ------------------- | --------------------------------------------------------------- | ----------- |
| 📓 Notebook         | All logic for RAG + FT (data prep, modeling, evaluation)        | ✅ Complete |
| 🖥️ Streamlit App    | UI with toggle for RAG/FT, query input, answer display          | ✅ Complete |
| 📊 PDF Report       | Screenshots, summary table, analysis                            | ⏳ Pending  |
| 🌐 Hosted App       | Streamlit Cloud or HuggingFace Space (optional but recommended) | ⏳ Pending  |
| 📁 requirements.txt | All Python libraries with versions for reproducibility          | ✅ Complete |
| 📦 Final ZIP        | Group\_<Number>\_RAG_vs_FT.zip with all code + assets           | ⏳ Pending  |

## 🛡️ Non-Functional Requirements (Met)

- Open-source only (no proprietary APIs)
- Guardrails required for safe/accurate answers
- Context limit compliance (chunked input for LLMs)
- UI response times:
  - <2s (RAG) with retrieval
  - <1s (FT) on test questions

## 📆 Project Timeline (Completed)

| Week   | Task                                            |
| ------ | ----------------------------------------------- |
| Week 1 | Data collection, Q/A generation, splitting      |
| Week 2 | RAG system: chunking, retrieval, basic UI       |
| Week 3 | Fine-tuning baseline, training, and testing     |
| Week 4 | Advanced techniques, guardrails, complete UI    |
| Week 5 | Evaluation, polish report, Streamlit deployment |

## 🔗 Tech Stack (Implemented)

| Layer       | Tool/Library                            |
| ----------- | --------------------------------------- |
| Embeddings  | all-MiniLM-L6-v2, E5-small-v2           |
| Retrieval   | FAISS, BM25 (Whoosh or sklearn)         |
| LLM         | DistilGPT2 / GPT-2 / MiniLM             |
| Fine-Tuning | HuggingFace Transformers, LoRA/Adapters |
| UI          | Streamlit                               |
| Hosting     | Streamlit Cloud / HuggingFace           |
| Runtime     | Colab / Local / GPU cluster             |

## ✅ Success Criteria

- [x] No data leakage between training/testing sets
- [x] UI works seamlessly and supports system switching
- [x] Guardrails are visibly functioning
- [x] Evaluation is well-documented and fair
- [x] Project runs successfully using only requirements.txt

## 📋 Project Tracking (Updated Status)

### Data Collection & Preprocessing

- [x] Collect financial statements (last 2 years)
- [x] Convert documents to plain text
- [x] Remove noise (headers, footers, page numbers)
- [x] Segment by section
- [x] Generate 50 Q/A pairs
- [x] Split into 40 training / 10 testing pairs

### RAG System Implementation

- [x] Implement document chunking (two sizes)
- [x] Set up embedding model
- [x] Implement dense retrieval index
- [x] Implement sparse retrieval index
- [x] Develop hybrid retrieval mechanism
- [x] Implement advanced technique
- [x] Set up answer generation with small LLM
- [x] Develop input-side guardrails
- [x] Develop output-side guardrails

### Fine-Tuned Model Implementation

- [x] Format Q/A pairs for fine-tuning
- [x] Select appropriate small open-source model
- [x] Perform pre-training evaluation
- [x] Set up fine-tuning process
- [x] Log fine-tuning parameters
- [x] Implement advanced fine-tuning method
- [x] Develop guardrails

### Evaluation System

- [x] Prepare test questions (high/low confidence, irrelevant)
- [x] Implement evaluation metrics tracking
- [x] Set up comparison framework
- [x] Document evaluation results

### UI Development

- [x] Create Streamlit app structure
- [x] Implement sidebar components
- [x] Implement main panel components
- [x] Add query validation
- [x] Add system toggle functionality
- [x] Implement context display toggle
- [x] Create evaluation results visualization
- [x] Ensure response time requirements are met

### Final Deliverables

- [x] Complete Jupyter notebook
- [x] Finalize Streamlit app
- [ ] Create PDF report
- [ ] Deploy app (optional)
- [x] Prepare requirements.txt
- [ ] Package final ZIP file
