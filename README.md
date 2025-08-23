# Financial Q&A System - Group 118

## Overview

A system comparing **RAG (Retrieval-Augmented Generation)** and **Fine-Tuned Language Models** for financial question answering.

**Advanced Techniques:**

- **RAG**: Cross-Encoder Re-ranking
- **Fine-Tuning**: Mixture-of-Experts with specialized routing

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python -m streamlit run app.py --server.port 8504 --server.headless true
   ```

3. **Open browser:** `http://localhost:8504`

4. **Complete setup wizard**

## How to Use

1. **Choose approach**: RAG or Fine-Tuned Model
2. **Ask questions**: _"What was the revenue in 2023?"_
3. **Compare results**: View answers, confidence scores, response times

## Key Features

- 155 Q&A pairs from financial reports
- Hybrid retrieval (BM25 + FAISS)
- PEFT fine-tuning with LoRA
- Real-time evaluation and comparison

---
