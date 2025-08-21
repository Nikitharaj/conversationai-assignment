import os
import json
import time
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

class FineTunedModel:
    """Class for using a fine-tuned language model for financial Q&A."""
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.model_path = Path(model_path)
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Check if this is a PEFT model
        is_peft = (self.model_path / "adapter_config.json").exists()
        
        if is_peft:
            # Load PEFT model
            peft_config = PeftConfig.from_pretrained(self.model_path)
            self.base_model_name = peft_config.base_model_name_or_path
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load PEFT adapter
            self.model = PeftModel.from_pretrained(self.base_model, self.model_path)
        else:
            # Load regular model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        
        # Set up generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Set default generation parameters
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
    
    def generate_answer(self, query: str) -> Tuple[str, float, float]:
        """
        Generate an answer based on the query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (answer, confidence_score, response_time)
        """
        # Format the query
        formatted_query = f"Question: {query}\nAnswer:"
        
        # Generate answer
        start_time = time.time()
        
        outputs = self.generator(
            formatted_query,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_return_sequences=1,
            return_full_text=False
        )
        
        # Extract the generated text
        answer = outputs[0]["generated_text"].strip()
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Calculate a simple confidence score based on output probabilities
        # This is a placeholder - real confidence would be more sophisticated
        confidence_score = 0.8  # Default confidence
        
        return answer, confidence_score, response_time
    
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
            'revenue', 'profit', 'income', 'earnings', 'sales', 'margin',
            'assets', 'liabilities', 'equity', 'cash flow', 'balance sheet',
            'financial', 'fiscal', 'quarter', 'annual', 'year', 'dividend',
            'stock', 'share', 'market', 'growth', 'decline', 'increase', 'decrease'
        ]
        
        # Check if any financial keyword is in the query
        is_financial = any(keyword.lower() in query.lower() for keyword in financial_keywords)
        
        if not is_financial:
            return "I can only answer questions related to financial information in the provided documents.", True
        
        return query, False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and generate an answer with guardrails.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Apply input-side guardrails
        filtered_query, is_filtered = self.filter_query(query)
        
        if is_filtered:
            return {
                "query": query,
                "answer": filtered_query,
                "confidence": 1.0,
                "response_time": 0.0,
                "is_filtered": True
            }
        
        # Generate answer
        answer, confidence, response_time = self.generate_answer(query)
        
        return {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "response_time": response_time,
            "is_filtered": False
        }


if __name__ == "__main__":
    # Example usage
    # model = FineTunedModel(model_path="../../models/fine_tuned")
    
    # Process a query
    # result = model.process_query("What was the revenue in Q2 2023?")
    # print(f"Answer: {result['answer']}")
    # print(f"Confidence: {result['confidence']:.2f}")
    # print(f"Response time: {result['response_time']:.3f}s")
