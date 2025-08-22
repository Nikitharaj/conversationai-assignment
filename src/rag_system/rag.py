"""
LangChain-based RAG implementation for Financial Q&A System.

This module provides a more stable and configurable RAG system using LangChain components.
"""

import os
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# LangChain imports - gracefully handle if not available
try:
    # Document processing
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Vector stores
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Retrievers
    from langchain.retrievers import BM25Retriever
    from langchain.retrievers.ensemble import EnsembleRetriever

    # LLM components (fallback to transformers if OpenAI not available)
    try:
        from langchain_openai import ChatOpenAI as LLM

        openai_available = True
    except (ImportError, ModuleNotFoundError):
        openai_available = False
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            transformers_available = True
        except (ImportError, ModuleNotFoundError):
            transformers_available = False

    # Chain components
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    langchain_available = True
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(
        f"LangChain not available: {e}. Install with 'pip install -r requirements_langchain.txt'"
    )
    langchain_available = False


class RAG:
    """LangChain-based RAG system for financial question answering."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: Optional[str] = "distilgpt2",
        retrieval_method: str = "hybrid",
        top_k: int = 3,
    ):
        """
        Initialize the LangChain RAG system.

        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the language model or None to disable
            retrieval_method: Retrieval method ("dense", "sparse", or "hybrid")
            top_k: Number of documents to retrieve
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.retrieval_method = retrieval_method
        self.top_k = top_k

        # Track initialization state
        self.is_initialized = False
        self.embeddings = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None

        # Initialize if LangChain is available
        if langchain_available:
            self._initialize_embeddings()
            self._initialize_llm()
        else:
            warnings.warn(
                "LangChain components not available. System will use fallback methods."
            )

    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            # Initialize HuggingFace embeddings with the specified model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print(f"Successfully loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            warnings.warn(f"Error loading embedding model: {e}")
            self.embeddings = None

    def _initialize_llm(self):
        """Initialize the language model."""
        # Skip if no model name provided
        if self.llm_model_name is None:
            print("LLM model disabled. Using retrieval-only mode.")
            return

        try:
            # Try OpenAI first if available (requires API key)
            if openai_available and os.environ.get("OPENAI_API_KEY"):
                self.llm = LLM(temperature=0.7, model="gpt-3.5-turbo")
                print("Using OpenAI for text generation")
            # Fall back to local transformers model
            elif transformers_available:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

                # Load model with memory-efficient settings
                model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name, low_cpu_mem_usage=True, torch_dtype="auto"
                )

                # Create text generation pipeline
                text_gen_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    device=-1,  # Use CPU
                )

                # Create HuggingFacePipeline LLM
                self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
                print(f"Using local model {self.llm_model_name} for text generation")
            else:
                warnings.warn("No LLM available. Will use retrieval-only mode.")
        except Exception as e:
            warnings.warn(f"Error initializing LLM: {e}")
            self.llm = None

    def build_indexes(self, documents: List[Dict[str, Any]]):
        """
        Build retrieval indexes from documents.

        Args:
            documents: List of document chunks with 'text' field
        """
        if not langchain_available:
            warnings.warn("LangChain not available. Cannot build indexes.")
            return

        try:
            # Extract text from documents
            texts = [doc.get("text", "") for doc in documents]
            metadatas = []

            # Extract metadata from documents
            for doc in documents:
                metadata = {k: v for k, v in doc.items() if k != "text"}
                metadatas.append(metadata)

            # Build dense index if embeddings are available
            if self.embeddings:
                # Create FAISS index
                vector_store = FAISS.from_texts(
                    texts=texts, embedding=self.embeddings, metadatas=metadatas
                )
                self.dense_retriever = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": self.top_k}
                )
                print("Dense retriever built successfully")

            # Build sparse (BM25) index
            self.sparse_retriever = BM25Retriever.from_texts(texts)
            self.sparse_retriever.k = self.top_k
            print("Sparse retriever built successfully")

            # Set up the appropriate retriever based on the retrieval method
            if self.retrieval_method == "dense" and self.dense_retriever:
                self.retriever = self.dense_retriever
                print("Using dense retrieval")
            elif self.retrieval_method == "sparse":
                self.retriever = self.sparse_retriever
                print("Using sparse retrieval")
            elif self.retrieval_method == "hybrid" and self.dense_retriever:
                # Create an ensemble retriever that combines dense and sparse
                self.retriever = EnsembleRetriever(
                    retrievers=[self.dense_retriever, self.sparse_retriever],
                    weights=[0.5, 0.5],
                )
                print("Using hybrid retrieval")
            else:
                # Fall back to sparse retrieval
                self.retriever = self.sparse_retriever
                print("Falling back to sparse retrieval")

            # Create QA chain if LLM is available
            if self.llm:
                # Define a custom prompt template for financial QA
                template = """
                You are a financial analyst assistant. Use the following context to answer the question.
                If you don't know the answer based on the context, just say "I don't have enough information to answer this question."
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                prompt = PromptTemplate(
                    template=template, input_variables=["context", "question"]
                )

                # Create the QA chain
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": prompt},
                )
                print("QA chain created successfully")

            self.is_initialized = True
            print("LangChain RAG system initialized successfully")

        except Exception as e:
            warnings.warn(f"Error building indexes: {e}")
            import traceback

            traceback.print_exc()

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and generate an answer.

        Args:
            query: User query

        Returns:
            Dictionary containing the answer and metadata
        """
        start_time = time.time()

        # Check if system is initialized
        if not self.is_initialized:
            return {
                "query": query,
                "answer": "The RAG system is not initialized with any documents yet.",
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "retrieved_chunks": [],
            }

        # Filter out non-financial queries
        financial_keywords = [
            "revenue",
            "profit",
            "income",
            "earnings",
            "sales",
            "margin",
            "assets",
            "liabilities",
            "equity",
            "cash flow",
            "balance sheet",
            "financial",
            "fiscal",
            "quarter",
            "annual",
            "year",
            "dividend",
            "stock",
            "share",
            "market",
            "growth",
            "decline",
            "increase",
            "decrease",
        ]

        is_financial = any(
            keyword.lower() in query.lower() for keyword in financial_keywords
        )

        if not is_financial:
            return {
                "query": query,
                "answer": "I can only answer questions related to financial information in the provided documents.",
                "confidence": 1.0,
                "response_time": time.time() - start_time,
                "is_filtered": True,
                "retrieved_chunks": [],
            }

        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(query)

            # Generate answer using QA chain if available
            if self.qa_chain:
                result = self.qa_chain.invoke({"query": query})
                answer = result["result"]
                confidence = 0.8  # Placeholder confidence score
            else:
                # Simple fallback if no LLM is available
                if retrieved_docs:
                    answer = f"Based on the retrieved information: {retrieved_docs[0].page_content[:500]}..."
                    confidence = 0.5
                else:
                    answer = (
                        "I couldn't find relevant information to answer your question."
                    )
                    confidence = 0.0

            # Format retrieved chunks for the response
            retrieved_chunks = []
            for i, doc in enumerate(retrieved_docs):
                chunk = {
                    "text": doc.page_content,
                    "score": 1.0 - (i * 0.1),  # Simple scoring based on rank
                    "method": self.retrieval_method,
                }
                if hasattr(doc, "metadata"):
                    chunk.update(doc.metadata)
                retrieved_chunks.append(chunk)

            response_time = time.time() - start_time

            return {
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "response_time": response_time,
                "is_filtered": False,
                "retrieved_chunks": retrieved_chunks,
            }

        except Exception as e:
            warnings.warn(f"Error processing query: {e}")
            import traceback

            traceback.print_exc()

            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "error": str(e),
                "retrieved_chunks": [],
            }

    def save(self, output_dir: Union[str, Path]):
        """
        Save the RAG system to disk.

        Args:
            output_dir: Directory to save the system
        """
        if not self.is_initialized:
            warnings.warn("System not initialized. Nothing to save.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        import json

        config = {
            "embedding_model_name": self.embedding_model_name,
            "llm_model_name": self.llm_model_name,
            "retrieval_method": self.retrieval_method,
            "top_k": self.top_k,
        }

        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save FAISS index if available
        if hasattr(self, "dense_retriever") and self.dense_retriever:
            vector_store = self.dense_retriever.vectorstore
            if hasattr(vector_store, "save_local"):
                vector_store_dir = output_dir / "vector_store"
                vector_store_dir.mkdir(exist_ok=True)
                vector_store.save_local(str(vector_store_dir))
                print(f"Vector store saved to {vector_store_dir}")

        print(f"LangChain RAG system configuration saved to {output_dir}")

    def load(self, input_dir: Union[str, Path]):
        """
        Load the RAG system from disk.

        Args:
            input_dir: Directory containing the saved system
        """
        if not langchain_available:
            warnings.warn("LangChain not available. Cannot load system.")
            return

        input_dir = Path(input_dir)

        # Load configuration
        import json

        try:
            with open(input_dir / "config.json", "r") as f:
                config = json.load(f)

            # Update configuration
            self.embedding_model_name = config.get(
                "embedding_model_name", self.embedding_model_name
            )
            self.llm_model_name = config.get("llm_model_name", self.llm_model_name)
            self.retrieval_method = config.get(
                "retrieval_method", self.retrieval_method
            )
            self.top_k = config.get("top_k", self.top_k)

            # Initialize components
            self._initialize_embeddings()
            self._initialize_llm()

            # Load FAISS index if available
            vector_store_dir = input_dir / "vector_store"
            if vector_store_dir.exists() and self.embeddings:
                try:
                    vector_store = FAISS.load_local(
                        str(vector_store_dir), self.embeddings
                    )
                    self.dense_retriever = vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": self.top_k}
                    )
                    print("Dense retriever loaded successfully")

                    # Set up the appropriate retriever based on the retrieval method
                    if self.retrieval_method == "dense":
                        self.retriever = self.dense_retriever
                        print("Using dense retrieval")
                    elif (
                        self.retrieval_method == "hybrid"
                        and hasattr(self, "sparse_retriever")
                        and self.sparse_retriever
                    ):
                        self.retriever = EnsembleRetriever(
                            retrievers=[self.dense_retriever, self.sparse_retriever],
                            weights=[0.5, 0.5],
                        )
                        print("Using hybrid retrieval")
                except Exception as e:
                    warnings.warn(f"Error loading vector store: {e}")

            # Create QA chain if LLM is available
            if self.llm and self.retriever:
                # Define a custom prompt template for financial QA
                template = """
                You are a financial analyst assistant. Use the following context to answer the question.
                If you don't know the answer based on the context, just say "I don't have enough information to answer this question."
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                prompt = PromptTemplate(
                    template=template, input_variables=["context", "question"]
                )

                # Create the QA chain
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": prompt},
                )
                print("QA chain created successfully")

            self.is_initialized = True
            print(f"LangChain RAG system loaded from {input_dir}")

        except Exception as e:
            warnings.warn(f"Error loading system: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    rag = LangChainRAG(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name=None,  # Disable LLM to avoid segmentation faults
        retrieval_method="sparse",
        top_k=3,
    )

    # Example documents
    documents = [
        {
            "text": "The company reported revenue of $10.5 million for Q2 2023.",
            "document": "financial_report.pdf",
        },
        {
            "text": "This represents a 15% increase from the same period last year.",
            "document": "financial_report.pdf",
        },
        {
            "text": "Operating expenses were $8.2 million, resulting in a profit margin of 21.9%.",
            "document": "financial_report.pdf",
        },
    ]

    # Build indexes
    rag.build_indexes(documents)

    # Process a query
    result = rag.process_query("What was the revenue in Q2 2023?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Response time: {result['response_time']:.3f}s")
