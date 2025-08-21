import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import random

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QAGenerator:
    """Class for generating Q&A pairs from financial documents."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the Q&A generator.
        
        Args:
            output_dir: Directory to save generated Q&A pairs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_qa_pairs(self, text_file: Union[str, Path], num_pairs: int = 10) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from a processed text file.
        
        Args:
            text_file: Path to the processed text file
            num_pairs: Number of Q&A pairs to generate
            
        Returns:
            List of dictionaries containing questions and answers
        """
        text_file = Path(text_file)
        
        if not text_file.exists():
            raise FileNotFoundError(f"File not found: {text_file}")
        
        # Read the processed text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Generate Q&A pairs
        qa_pairs = []
        
        # Extract financial metrics and generate questions
        qa_pairs.extend(self._generate_financial_metric_qa(text))
        
        # Extract dates and generate questions
        qa_pairs.extend(self._generate_date_based_qa(text))
        
        # Extract comparisons and generate questions
        qa_pairs.extend(self._generate_comparison_qa(text))
        
        # Limit to the requested number of pairs
        qa_pairs = qa_pairs[:num_pairs]
        
        # Save the generated Q&A pairs
        output_file = self.output_dir / f"{text_file.stem}_qa_pairs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2)
        
        return qa_pairs
    
    def _generate_financial_metric_qa(self, text: str) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs about financial metrics.
        
        Args:
            text: Processed document text
            
        Returns:
            List of Q&A pairs about financial metrics
        """
        qa_pairs = []
        
        # Look for revenue/sales information
        revenue_patterns = [
            r'(?i)revenue[s]?\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?',
            r'(?i)sales\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?',
            r'(?i)total\s+revenue[s]?\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?'
        ]
        
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Generate a question about revenue
                qa_pairs.append({
                    "question": "What was the revenue reported in the financial statement?",
                    "answer": f"The revenue was ${match}.",
                    "source": "revenue_pattern"
                })
        
        # Look for profit/income information
        profit_patterns = [
            r'(?i)net\s+(?:income|profit|earnings)\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?',
            r'(?i)operating\s+(?:income|profit)\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?',
            r'(?i)gross\s+(?:profit|margin)\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?'
        ]
        
        for pattern in profit_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Generate a question about profit
                qa_pairs.append({
                    "question": "What was the net income/profit reported?",
                    "answer": f"The net income/profit was ${match}.",
                    "source": "profit_pattern"
                })
        
        # Look for assets/liabilities information
        asset_patterns = [
            r'(?i)total\s+assets\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?',
            r'(?i)total\s+liabilities\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?',
            r'(?i)shareholders[\'']?\s+equity\s+(?:was|of|:)\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?'
        ]
        
        for pattern in asset_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Generate a question about assets/liabilities
                qa_pairs.append({
                    "question": "What were the total assets reported in the financial statement?",
                    "answer": f"The total assets were ${match}.",
                    "source": "asset_pattern"
                })
        
        return qa_pairs
    
    def _generate_date_based_qa(self, text: str) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs about dates and time periods.
        
        Args:
            text: Processed document text
            
        Returns:
            List of Q&A pairs about dates and time periods
        """
        qa_pairs = []
        
        # Look for fiscal year/quarter information
        date_patterns = [
            r'(?i)fiscal\s+year\s+(?:ended|ending)\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)quarter\s+ended\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)(?:for|in)\s+(?:the|)\s+year\s+(\d{4})',
            r'(?i)(?:Q[1-4]|first|second|third|fourth)\s+quarter\s+(?:of|)\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Generate a question about the time period
                qa_pairs.append({
                    "question": "What period does this financial statement cover?",
                    "answer": f"The financial statement covers the period ending {match}.",
                    "source": "date_pattern"
                })
        
        return qa_pairs
    
    def _generate_comparison_qa(self, text: str) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs about year-over-year or quarter-over-quarter comparisons.
        
        Args:
            text: Processed document text
            
        Returns:
            List of Q&A pairs about comparisons
        """
        qa_pairs = []
        
        # Look for comparison information
        comparison_patterns = [
            r'(?i)increased\s+by\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?\s+(?:or\s+)?([\d\.]+)\%?',
            r'(?i)decreased\s+by\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?\s+(?:or\s+)?([\d\.]+)\%?',
            r'(?i)growth\s+of\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?\s+(?:or\s+)?([\d\.]+)\%?',
            r'(?i)decline\s+of\s+\$?([\d,\.]+)\s+(?:million|billion|M|B)?\s+(?:or\s+)?([\d\.]+)\%?'
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Extract the context (sentence containing the match)
                sentences = sent_tokenize(text)
                context = ""
                for sentence in sentences:
                    if any(m in sentence for m in match):
                        context = sentence
                        break
                
                # Generate a question about the comparison
                qa_pairs.append({
                    "question": "How did the financial performance change compared to the previous period?",
                    "answer": context,
                    "source": "comparison_pattern"
                })
        
        return qa_pairs
    
    def split_train_test(self, qa_pairs: List[Dict[str, str]], test_ratio: float = 0.2) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split Q&A pairs into training and testing sets.
        
        Args:
            qa_pairs: List of Q&A pairs
            test_ratio: Ratio of test samples (0.0 to 1.0)
            
        Returns:
            Tuple of (training_pairs, testing_pairs)
        """
        # Shuffle the Q&A pairs
        random.shuffle(qa_pairs)
        
        # Calculate the split point
        split_idx = int(len(qa_pairs) * (1 - test_ratio))
        
        # Split the data
        train_pairs = qa_pairs[:split_idx]
        test_pairs = qa_pairs[split_idx:]
        
        return train_pairs, test_pairs
    
    def save_train_test_split(self, train_pairs: List[Dict[str, str]], test_pairs: List[Dict[str, str]], base_name: str):
        """
        Save the training and testing Q&A pairs to files.
        
        Args:
            train_pairs: List of training Q&A pairs
            test_pairs: List of testing Q&A pairs
            base_name: Base name for the output files
        """
        # Save training pairs
        train_file = self.output_dir / f"{base_name}_train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_pairs, f, indent=2)
        
        # Save testing pairs
        test_file = self.output_dir / f"{base_name}_test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_pairs, f, indent=2)
    
    def batch_generate(self, processed_dir: Union[str, Path], num_pairs_per_file: int = 10) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs from all text files in a directory.
        
        Args:
            processed_dir: Directory containing processed text files
            num_pairs_per_file: Number of Q&A pairs to generate per file
            
        Returns:
            Combined list of all generated Q&A pairs
        """
        processed_dir = Path(processed_dir)
        all_qa_pairs = []
        
        for text_file in processed_dir.glob('*.txt'):
            try:
                qa_pairs = self.generate_qa_pairs(text_file, num_pairs_per_file)
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                print(f"Error generating Q&A pairs from {text_file}: {e}")
        
        # Save all Q&A pairs combined
        if all_qa_pairs:
            combined_file = self.output_dir / "all_qa_pairs.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, indent=2)
            
            # Create train/test split (40 train, 10 test)
            train_size = min(40, int(len(all_qa_pairs) * 0.8))
            test_size = min(10, len(all_qa_pairs) - train_size)
            
            # Ensure we have enough pairs
            if train_size + test_size <= len(all_qa_pairs):
                # Shuffle and split
                random.shuffle(all_qa_pairs)
                train_pairs = all_qa_pairs[:train_size]
                test_pairs = all_qa_pairs[train_size:train_size + test_size]
                
                # Save the split
                self.save_train_test_split(train_pairs, test_pairs, "financial_qa")
        
        return all_qa_pairs


if __name__ == "__main__":
    # Example usage
    generator = QAGenerator(output_dir="../../data/qa_pairs")
    
    # Generate Q&A pairs from a single file
    # generator.generate_qa_pairs("../../data/processed/example.txt", num_pairs=10)
    
    # Generate Q&A pairs from all files in a directory
    # generator.batch_generate("../../data/processed", num_pairs_per_file=5)
