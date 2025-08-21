import os
import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class AnswerGenerator:
    """Class for generating answers from retrieved document chunks."""

    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the language model to use
        """
        self.model_name = model_name

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set up generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Set default generation parameters
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50

    def format_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Format a prompt for the language model using the query and retrieved chunks.

        Args:
            query: User query
            chunks: Retrieved document chunks

        Returns:
            Formatted prompt
        """
        # Extract text from chunks
        chunk_texts = [chunk["chunk"]["text"] for chunk in chunks]

        # Combine chunks
        combined_context = "\n\n".join(chunk_texts)

        # Format the prompt
        prompt = f"""Context information:
{combined_context}

Question: {query}

Answer:"""

        return prompt

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

        # Format the prompt
        prompt = self.format_prompt(query, chunks)

        # Generate answer
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_return_sequences=1,
            return_full_text=False,
        )

        # Extract the generated text
        answer = outputs[0]["generated_text"].strip()

        # Calculate response time
        response_time = time.time() - start_time

        # Calculate a simple confidence score based on chunk relevance
        # This is a placeholder - real confidence would be more sophisticated
        avg_chunk_score = (
            sum(chunk["score"] for chunk in chunks) / len(chunks) if chunks else 0
        )
        confidence_score = min(avg_chunk_score, 1.0)  # Normalize to [0, 1]

        return answer, confidence_score, response_time

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
        # Extract text from chunks to check against
        chunk_texts = [chunk["chunk"]["text"].lower() for chunk in chunks]
        combined_context = " ".join(chunk_texts)

        # Check if the answer contains specific numbers or dates not in the context
        # This is a simple heuristic - real hallucination detection would be more sophisticated
        import re

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
        ]

        # Check if any financial keyword is in the query
        is_financial = any(
            keyword.lower() in query.lower() for keyword in financial_keywords
        )

        if not is_financial:
            return (
                "I can only answer questions related to financial information in the provided documents.",
                True,
            )

        return query, False

    def save_model(self, output_dir: Union[str, Path]):
        """
        Save the model and tokenizer to disk.

        Args:
            output_dir: Directory to save the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, input_dir: Union[str, Path]):
        """
        Load the model and tokenizer from disk.

        Args:
            input_dir: Directory containing the saved model
        """
        input_dir = Path(input_dir)

        self.model = AutoModelForCausalLM.from_pretrained(input_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)

        # Update the generator pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )


if __name__ == "__main__":
    # Example usage
    generator = AnswerGenerator(model_name="distilgpt2")

    # Example chunks
    # chunks = [
    #     {"chunk": {"text": "The company reported revenue of $10.5 million for Q2 2023."}, "score": 0.9, "method": "dense"},
    #     {"chunk": {"text": "This represents a 15% increase from the same period last year."}, "score": 0.8, "method": "bm25"}
    # ]

    # Generate answer
    # answer, confidence, response_time = generator.generate_answer("What was the revenue in Q2 2023?", chunks)
    # print(f"Answer: {answer}")
    # print(f"Confidence: {confidence:.2f}")
    # print(f"Response time: {response_time:.3f}s")
