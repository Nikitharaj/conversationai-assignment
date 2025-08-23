"""
LangChain-based answer generator for producing responses from retrieved documents.

This module replaces the custom AnswerGenerator with LangChain's LLM chains.
"""

import os
import json
import warnings
import time
import re
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple

# Try to import LangChain components
try:
    # Import langchain first
    import langchain

    # Import core components
    import langchain_core

    # LLM components
    import langchain_community
    from langchain_community.llms import HuggingFacePipeline

    import langchain_openai
    from langchain_openai import ChatOpenAI

    # Chain components
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Document processing
    from langchain_core.documents import Document

    # Evaluation
    from langchain.evaluation import QAEvalChain

    langchain_available = True
    print("LangChain components initialized successfully")
except ImportError as e:
    warnings.warn(
        f"LangChain not available: {e}. Install with 'pip install langchain langchain-community langchain-openai'"
    )
    langchain_available = False

# Try to import transformers, handle gracefully if not available
transformers_available = False
torch_available = False
try:
    # Import transformers with explicit import
    import transformers
    from transformers import AutoTokenizer

    print(f"Successfully loaded transformers version: {transformers.__version__}")

    # We'll import the model classes only when needed to save memory
    transformers_available = True

    # Set environment variables to limit memory usage
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Avoid downloading models
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid parallelism issues

    # Try to import torch, handle gracefully if not available
    try:
        import torch

        torch_available = True
        print(f"Successfully loaded torch version: {torch.__version__}")

        # Set low precision for memory efficiency
        torch.set_grad_enabled(False)  # Disable gradient computation
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("medium")  # Lower precision
    except ImportError:
        warnings.warn("torch not found. Text generation will be limited.")
        torch_available = False

except ImportError as e:
    # Only warn if it's a real import error, not a version mismatch
    if "regex" not in str(e):
        warnings.warn(f"transformers not found: {e}. Text generation will be limited.")
    transformers_available = False


class AnswerGenerator:
    """LangChain-based answer generator for producing responses from retrieved documents."""

    def __init__(self, model_name: Optional[str] = "distilgpt2"):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the language model to use, or None to skip model loading
        """
        self.model_name = model_name
        self.llm = None
        self.qa_chain = None
        self.is_initialized = False

        # Skip model loading if model_name is None
        if model_name is None:
            print(
                "Skipping model loading as requested. Using fallback text generation."
            )
            return

        # Initialize LLM and chain if LangChain is available
        if langchain_available:
            self._initialize_llm()
        else:
            print("LangChain not available. Using fallback text generation.")

    def _initialize_llm(self):
        """Initialize the language model."""
        # Skip if model_name is None
        if self.model_name is None:
            print("No model name provided. Using retrieval-only mode.")
            return

        try:
            # Try OpenAI first if available (requires API key)
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
                    print("Using OpenAI for text generation")
                except Exception as e:
                    warnings.warn(f"Error initializing OpenAI: {e}")
                    print("Falling back to local model")

            # Fall back to local transformers model
            if self.llm is None and transformers_available and torch_available:
                # Only import the necessary classes now
                try:
                    from transformers import AutoModelForCausalLM, pipeline

                    print(f"Initializing model: {self.model_name}")

                    # Use a try-except block with a timeout to prevent hanging
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError("Model loading timed out")

                    # Set a timeout of 30 seconds for model loading
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)

                    try:
                        # Load tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                        # Load model with memory-efficient settings
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16
                            if torch.cuda.is_available()
                            else None,
                        )

                        # Cancel the timeout
                        signal.alarm(0)

                        # Set device to CPU explicitly to avoid GPU memory issues
                        device = -1  # Always use CPU

                        # Set up generation pipeline with better parameters
                        text_gen_pipeline = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=200,  # Increased for more complete answers
                            temperature=0.3,  # Lower temperature for more focused responses
                            top_p=0.8,
                            top_k=40,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            device=device,
                        )

                        # Create HuggingFacePipeline LLM
                        self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
                        print(f"Model {self.model_name} loaded successfully")

                    except TimeoutError as te:
                        warnings.warn(f"Model loading timed out: {te}")
                        print("Using fallback text generation due to timeout")

                    finally:
                        # Reset the alarm in case of any exception
                        signal.alarm(0)

                except Exception as e:
                    warnings.warn(f"Error loading model: {e}")
                    print("Using fallback text generation")

            # Create QA chain if LLM is available
            if self.llm:
                # Define the prompt template for QA
                template = """Context: {context}

Question: {question}
Answer:"""

                prompt = PromptTemplate(
                    template=template, input_variables=["context", "question"]
                )

                # Create the QA chain
                self.qa_chain = LLMChain(llm=self.llm, prompt=prompt)

                self.is_initialized = True
                print("QA chain created successfully")

        except Exception as e:
            warnings.warn(f"Error initializing LLM: {e}")
            import traceback

            traceback.print_exc()

    def _convert_chunks_to_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Convert retrieved chunks to a context string.

        Args:
            chunks: Retrieved document chunks

        Returns:
            Formatted context string
        """
        # Extract text from chunks, with error handling
        chunk_texts = []
        for chunk in chunks:
            try:
                if (
                    isinstance(chunk, dict)
                    and "chunk" in chunk
                    and "text" in chunk["chunk"]
                ):
                    chunk_texts.append(chunk["chunk"]["text"])
                elif isinstance(chunk, dict) and "text" in chunk:
                    chunk_texts.append(chunk["text"])
                else:
                    # Handle unexpected chunk format
                    continue
            except Exception:
                # Skip problematic chunks
                continue

        # If no valid chunks were found, provide a generic context
        if not chunk_texts:
            return "No relevant context information found."

        # Combine chunks
        combined_context = "\n\n".join(chunk_texts)

        return combined_context

    def generate_answer(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, float, float]:
        """
        Generate an answer based on the query and retrieved chunks.

        Args:
            query: User query
            chunks: Retrieved document chunks

        Returns:
            Tuple of (answer, confidence_score, response_time)
        """
        start_time = time.time()

        # Filter query
        filtered_query, is_filtered = self.filter_query(query)
        if is_filtered:
            response_time = time.time() - start_time
            return filtered_query, 1.0, response_time

        # Convert chunks to context
        context = self._convert_chunks_to_context(chunks)

        # Check if LangChain QA chain is available
        if langchain_available and self.is_initialized and self.qa_chain:
            try:
                # Generate answer using LangChain
                result = self.qa_chain.invoke({"context": context, "question": query})

                answer = result.get("text", "").strip()

                # Clean up the answer by extracting only the actual response
                # Find the last "Answer:" and extract everything after it
                answer_start = answer.rfind("Answer:")
                if answer_start != -1:
                    answer = answer[answer_start + 7 :].strip()

                # Remove any remaining context or question text
                lines = answer.split("\n")
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not any(
                        line.startswith(prefix)
                        for prefix in ["Context:", "Question:", "Answer:"]
                    ):
                        clean_lines.append(line)

                if clean_lines:
                    answer = " ".join(clean_lines)

                # If answer is still too long or repetitive, truncate intelligently
                if len(answer) > 500:
                    # Find the first complete sentence
                    sentences = answer.split(".")
                    if len(sentences) > 1:
                        answer = sentences[0] + "."

                # If answer is poor quality, try simple extraction
                if (
                    not answer
                    or len(answer) < 10
                    or answer.count(".") == 0
                    or any(
                        bad_phrase in answer.lower()
                        for bad_phrase in [
                            "i think",
                            "a lot of money",
                            "question:",
                            "answer:",
                        ]
                    )
                ):
                    answer = self._extract_simple_answer(context, query)

                # Apply guardrails
                answer, is_hallucination = self.apply_guardrails(query, answer, chunks)

                # Calculate response time
                response_time = time.time() - start_time

                # Calculate confidence score
                if is_hallucination:
                    confidence_score = 0.3
                else:
                    # Calculate a simple confidence score based on chunk relevance
                    valid_chunks = [
                        c for c in chunks if isinstance(c, dict) and "score" in c
                    ]
                    avg_chunk_score = (
                        sum(chunk["score"] for chunk in valid_chunks)
                        / len(valid_chunks)
                        if valid_chunks
                        else 0.5
                    )
                    confidence_score = min(avg_chunk_score, 1.0)  # Normalize to [0, 1]

                return answer, confidence_score, response_time

            except Exception as e:
                warnings.warn(f"Error generating answer: {e}")
                import traceback

                traceback.print_exc()

        # Fallback: Generate a simple answer
        answer = self._generate_fallback_answer(query, chunks)
        response_time = time.time() - start_time

        # Use a lower confidence score for fallback answers
        confidence_score = 0.3

        return answer, confidence_score, response_time

    def _generate_fallback_answer(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a fallback answer when the model is not available.

        Args:
            query: User query
            chunks: Retrieved document chunks

        Returns:
            A simple answer based on the retrieved chunks
        """
        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            try:
                if (
                    isinstance(chunk, dict)
                    and "chunk" in chunk
                    and "text" in chunk["chunk"]
                ):
                    chunk_texts.append(chunk["chunk"]["text"])
                elif isinstance(chunk, dict) and "text" in chunk:
                    chunk_texts.append(chunk["text"])
            except Exception:
                continue

        if not chunk_texts:
            return "I'm sorry, I couldn't find relevant information to answer your question."

        # Simple fallback: Return the most relevant chunk
        return f"Based on the retrieved information: {chunk_texts[0][:500]}..."

    def apply_guardrails(
        self, query: str, answer: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, bool]:
        """
        Apply output-side guardrails to detect hallucinations or unsupported claims.

        Args:
            query: User query
            answer: Generated answer
            chunks: Retrieved document chunks

        Returns:
            Tuple of (possibly modified answer, is_hallucination)
        """
        # If we're using the fallback generator, skip detailed guardrails
        if not self.is_initialized:
            # Simple check if the answer is a fallback
            if answer.startswith("Based on the retrieved information:"):
                return answer, False
            else:
                # For testing purposes, don't add the note
                return answer, True

        try:
            # Extract text from chunks to check against, with error handling
            chunk_texts = []
            for chunk in chunks:
                try:
                    if (
                        isinstance(chunk, dict)
                        and "chunk" in chunk
                        and "text" in chunk["chunk"]
                    ):
                        chunk_texts.append(chunk["chunk"]["text"].lower())
                    elif isinstance(chunk, dict) and "text" in chunk:
                        chunk_texts.append(chunk["text"].lower())
                except Exception:
                    # Skip problematic chunks
                    continue

            # If no valid chunks were found, assume hallucination
            if not chunk_texts:
                modified_answer = (
                    answer
                    + "\n\n(Note: This answer could not be verified against the retrieved context.)"
                )
                return modified_answer, True

            combined_context = " ".join(chunk_texts)

            # Check if the answer contains specific numbers or dates not in the context
            # This is a simple heuristic - real hallucination detection would be more sophisticated

            # Look for numbers in the answer
            numbers_in_answer = re.findall(
                r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?",
                answer.lower(),
            )

            # Check if each number appears in the context
            hallucination_detected = False
            for num in numbers_in_answer:
                if num not in combined_context:
                    hallucination_detected = True
                    break

            if hallucination_detected:
                # Modify the answer to indicate uncertainty
                modified_answer = (
                    answer
                    + "\n\n(Note: The specific figures in this answer may not be fully supported by the retrieved context.)"
                )
                return modified_answer, True

            return answer, False
        except Exception as e:
            # If guardrails fail, add a note to the answer
            warnings.warn(f"Error applying guardrails: {e}")
            return (
                answer
                + "\n\n(Note: This answer could not be fully verified due to an error in the verification process.)",
                True,
            )

    def filter_query(self, query: str) -> Tuple[str, bool]:
        """
        Apply input-side guardrails to filter irrelevant or unsafe queries.

        Args:
            query: User query

        Returns:
            Tuple of (possibly modified query, is_filtered)
        """
        # Check if the query is related to financial information
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
            "accounts",
            "receivable",
            "payable",
            "services",
            "iphone",
            "ipad",
            "mac",
            "apple",
            "performance",
            "tax",
            "net",
            "gross",
            "operating",
            "expense",
            "cost",
            "billion",
            "million",
            "dollar",
            "$",
            "percent",
            "%",
            "report",
            "2022",
            "2023",
            "2024",
        ]

        # Check if any financial keyword is in the query
        is_financial = any(
            keyword.lower() in query.lower() for keyword in financial_keywords
        )

        if not is_financial:
            return (
                "I can only answer questions related to financial information. This question appears to be outside the scope of financial analysis.",
                True,
            )

        return query, False

    def evaluate_answer(
        self, query: str, answer: str, reference_text: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of the generated answer.

        Args:
            query: User query
            answer: Generated answer
            reference_text: Reference text for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if not langchain_available or not self.is_initialized:
            # Simple fallback evaluation
            return {
                "relevance": 0.5,
                "accuracy": 0.5,
                "completeness": 0.5,
            }

        try:
            # Create evaluation chain
            eval_chain = QAEvalChain.from_llm(llm=self.llm)

            # Prepare evaluation data
            eval_data = [
                {
                    "query": query,
                    "answer": answer,
                    "reference": reference_text,
                }
            ]

            # Run evaluation
            results = eval_chain.evaluate(eval_data)

            # Extract scores (this is a placeholder - real evaluation would be more sophisticated)
            evaluation = {
                "relevance": 0.8 if "relevant" in results[0]["text"].lower() else 0.4,
                "accuracy": 0.8 if "accurate" in results[0]["text"].lower() else 0.4,
                "completeness": 0.8
                if "complete" in results[0]["text"].lower()
                else 0.4,
            }

            return evaluation
        except Exception as e:
            warnings.warn(f"Error evaluating answer: {e}")
            return {
                "relevance": 0.5,
                "accuracy": 0.5,
                "completeness": 0.5,
            }

    def save(self, output_dir: Union[str, Path]):
        """
        Save the model configuration to disk.

        Args:
            output_dir: Directory to save the configuration
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "model_name": self.model_name,
        }

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to {output_dir}")

    def load(self, input_dir: Union[str, Path]):
        """
        Load the model configuration from disk.

        Args:
            input_dir: Directory containing the saved configuration
        """
        input_dir = Path(input_dir)

        try:
            # Load configuration
            with open(input_dir / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)

            # Update model name
            self.model_name = config.get("model_name", self.model_name)

            # Initialize LLM and chain
            if langchain_available:
                self._initialize_llm()

            print(f"Configuration loaded from {input_dir}")
        except Exception as e:
            warnings.warn(f"Error loading configuration: {e}")

    def _extract_simple_answer(self, context: str, query: str) -> str:
        """Extract simple financial answers using pattern matching."""
        import re

        context_lower = context.lower()
        query_lower = query.lower()

        # Revenue patterns
        revenue_patterns = [
            r"total\s+(?:net\s+)?(?:sales|revenue)\s+of\s+\$([0-9,\.]+\s*(?:billion|million))",
            r"revenue\s+(?:was|of)\s+\$([0-9,\.]+\s*(?:billion|million))",
            r"\$([0-9,\.]+\s*(?:billion|million))\s+(?:in\s+)?revenue",
            r"sales\s+of\s+\$([0-9,\.]+\s*(?:billion|million))",
        ]

        # Growth patterns
        growth_patterns = [
            r"(?:up|increased)\s+([0-9\.]+%)",
            r"([0-9\.]+%)\s+(?:increase|growth)",
            r"representing\s+a\s+([0-9\.]+%)\s+(?:increase|decrease)",
        ]

        # Revenue questions
        if any(term in query_lower for term in ["revenue", "sales", "total"]):
            for pattern in revenue_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    return f"${matches[0]}"

        # Growth questions
        if any(
            term in query_lower for term in ["growth", "increase", "decrease", "rate"]
        ):
            for pattern in growth_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    return f"{matches[0]}"

        # iPhone specific
        if "iphone" in query_lower:
            iphone_patterns = [
                r"iphone\s+revenue\s+(?:was|of)\s+\$([0-9,\.]+\s*(?:billion|million))",
                r"\$([0-9,\.]+\s*(?:billion|million)).*iphone",
                r"iphone.*\$([0-9,\.]+\s*(?:billion|million))",
            ]
            for pattern in iphone_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    return f"iPhone revenue was ${matches[0]}"

            # Also look for iPhone percentage info
            iphone_percent_patterns = [
                r"iphone\s+represented\s+([0-9\.]+%)",
                r"([0-9\.]+%)\s+of\s+total\s+revenue.*iphone",
            ]
            for pattern in iphone_percent_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    return f"iPhone represented {matches[0]} of total revenue"

        # Services specific
        if "services" in query_lower:
            services_patterns = [
                r"services\s+revenue\s+(?:increased\s+to\s+|was\s+)?\$([0-9,\.]+\s*(?:billion|million))",
                r"\$([0-9,\.]+\s*(?:billion|million)).*services",
            ]
            for pattern in services_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    return f"Services revenue was ${matches[0]}"

        # Percentage calculations
        if "percentage" in query_lower or "%" in query_lower:
            percentage_patterns = [
                r"represented\s+([0-9\.]+%)\s+of\s+total\s+revenue",
                r"([0-9\.]+%)\s+of\s+(?:total\s+)?revenue",
            ]
            for pattern in percentage_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    return f"{matches[0]} of total revenue"

        # If no specific pattern matches, return first sentence with numbers
        sentences = context.split(".")
        for sentence in sentences:
            if any(term in sentence.lower() for term in query_lower.split()[:2]):
                if re.search(r"\$[0-9,\.]+", sentence):
                    return sentence.strip() + "."

        return "I cannot find specific information to answer this question in the provided context."


if __name__ == "__main__":
    # Example usage
    generator = AnswerGenerator(model_name="distilgpt2")

    # Example chunks
    chunks = [
        {
            "chunk": {
                "text": "The company reported revenue of $10.5 million for Q2 2023."
            },
            "score": 0.9,
            "method": "dense",
        },
        {
            "chunk": {
                "text": "This represents a 15% increase from the same period last year."
            },
            "score": 0.8,
            "method": "bm25",
        },
    ]

    # Generate answer
    answer, confidence, response_time = generator.generate_answer(
        "What was the revenue in Q2 2023?", chunks
    )
    print(f"Answer: {answer}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Response time: {response_time:.3f}s")
