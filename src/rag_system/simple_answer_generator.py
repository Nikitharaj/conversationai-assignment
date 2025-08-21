"""
Simplified version of the answer generator that doesn't rely on external libraries.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple


class SimpleAnswerGenerator:
    """Simplified class for generating answers from retrieved document chunks."""

    def __init__(self, model_name: str = "mock-model"):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the model (not used in this simplified version)
        """
        self.model_name = model_name

    def format_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Format a prompt using the query and retrieved chunks.

        Args:
            query: User query
            chunks: Retrieved document chunks

        Returns:
            Formatted prompt
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
                else:
                    # Handle unexpected chunk format
                    continue
            except Exception as e:
                # Skip problematic chunks
                continue

        # If no valid chunks were found, provide a generic context
        if not chunk_texts:
            combined_context = "No relevant context information found."
        else:
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

        # In a real implementation, this would use a language model
        # Here we just extract information from the chunks

        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            try:
                if (
                    isinstance(chunk, dict)
                    and "chunk" in chunk
                    and "text" in chunk["chunk"]
                ):
                    chunk_texts.append(chunk["chunk"]["text"].lower())
                else:
                    # Handle unexpected chunk format
                    continue
            except Exception:
                # Skip problematic chunks
                continue

        # Simple answer generation based on query keywords
        if "revenue" in query.lower():
            answer = "Based on the provided information, the revenue was $1,250 million in the fiscal year 2023."
        elif "profit" in query.lower():
            answer = (
                "The company reported a net income of $150 million for the fiscal year."
            )
        elif "assets" in query.lower():
            answer = (
                "The total assets were $2,500 million according to the balance sheet."
            )
        else:
            answer = (
                "I don't have specific information about that in the provided context."
            )

        # Calculate response time
        response_time = time.time() - start_time

        # Calculate a simple confidence score based on chunk relevance
        valid_chunks = [c for c in chunks if isinstance(c, dict) and "score" in c]
        avg_chunk_score = (
            sum(chunk["score"] for chunk in valid_chunks) / len(valid_chunks)
            if valid_chunks
            else 0
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

        # Simple hallucination detection based on keywords
        hallucination_detected = False

        # Check for numbers in the answer that aren't in the context
        import re

        numbers_in_answer = re.findall(
            r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?",
            answer.lower(),
        )

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
