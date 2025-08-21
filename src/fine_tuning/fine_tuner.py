import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FineTuner:
    """Class for fine-tuning language models on financial Q&A data."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: Union[str, Path] = "fine_tuned_model",
        use_peft: bool = True,
    ):
        """
        Initialize the fine-tuner.

        Args:
            model_name: Name of the base model to fine-tune
            output_dir: Directory to save the fine-tuned model
            use_peft: Whether to use Parameter-Efficient Fine-Tuning (PEFT)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_peft = use_peft

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Apply PEFT if requested
        if use_peft:
            self.apply_peft()

        # Training parameters
        self.max_length = 512
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.num_epochs = 3

        # Track training metrics
        self.training_metrics = {}

    def apply_peft(self):
        """Apply Parameter-Efficient Fine-Tuning (PEFT) to the model."""
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Target attention modules
        )

        # Prepare model for k-bit training if using 8-bit or 4-bit quantization
        if "8bit" in self.model_name or "4bit" in self.model_name:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)

        logger.info("Applied PEFT (LoRA) to the model")

    def prepare_dataset(self, qa_file: Union[str, Path]) -> Dataset:
        """
        Prepare a dataset from Q&A pairs.

        Args:
            qa_file: Path to a JSON file containing Q&A pairs

        Returns:
            Hugging Face Dataset
        """
        qa_file = Path(qa_file)

        # Load Q&A pairs
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

        # Format data for fine-tuning
        formatted_data = []

        for pair in qa_pairs:
            # Format as a prompt-completion pair
            formatted_text = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
            formatted_data.append({"text": formatted_text})

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        return tokenized_dataset

    def fine_tune(
        self, train_file: Union[str, Path], eval_file: Optional[Union[str, Path]] = None
    ):
        """
        Fine-tune the model on Q&A data.

        Args:
            train_file: Path to a JSON file containing training Q&A pairs
            eval_file: Optional path to a JSON file containing evaluation Q&A pairs
        """
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_file)
        eval_dataset = self.prepare_dataset(eval_file) if eval_file else None

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_steps=200,
            save_steps=200,
            warmup_steps=100,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",  # Disable wandb, tensorboard, etc.
        )

        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're not doing masked language modeling
        )

        # Set up trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Log hardware info
        device_info = f"Device: {self.model.device}"
        if torch.cuda.is_available():
            device_info += f", GPU: {torch.cuda.get_device_name(0)}"
        logger.info(device_info)

        # Log training parameters
        logger.info(f"Training with parameters:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  PEFT: {self.use_peft}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")

        # Train the model
        logger.info("Starting fine-tuning...")
        train_result = trainer.train()

        # Save training metrics
        self.training_metrics = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0
            ),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "epoch": train_result.metrics.get("epoch", 0),
            "model_name": self.model_name,
            "use_peft": self.use_peft,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "device": device_info,
        }

        # Save metrics
        with open(
            self.output_dir / "training_metrics.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.training_metrics, f, indent=2)

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info(f"Fine-tuning complete. Model saved to {self.output_dir}")

    def evaluate(self, test_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model on test Q&A pairs.

        Args:
            test_file: Path to a JSON file containing test Q&A pairs

        Returns:
            Dictionary of evaluation metrics
        """
        test_file = Path(test_file)

        # Load test Q&A pairs
        with open(test_file, "r", encoding="utf-8") as f:
            test_pairs = json.load(f)

        # Set up evaluation pipeline
        from transformers import pipeline

        qa_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Evaluate each test pair
        results = []
        total_time = 0
        correct_count = 0

        for pair in test_pairs:
            # Format the question
            question = f"Question: {pair['question']}\nAnswer:"

            # Generate answer
            import time

            start_time = time.time()

            output = qa_pipeline(
                question,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                return_full_text=False,
            )

            end_time = time.time()
            response_time = end_time - start_time
            total_time += response_time

            # Extract the generated answer
            generated_answer = output[0]["generated_text"].strip()

            # Calculate a simple accuracy score (placeholder)
            # In a real implementation, this would use NLP metrics
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(
                None, generated_answer.lower(), pair["answer"].lower()
            ).ratio()
            is_correct = similarity > 0.5

            if is_correct:
                correct_count += 1

            # Store result
            results.append(
                {
                    "question": pair["question"],
                    "ground_truth": pair["answer"],
                    "generated": generated_answer,
                    "response_time": response_time,
                    "similarity": similarity,
                    "is_correct": is_correct,
                }
            )

        # Calculate overall metrics
        accuracy = correct_count / len(test_pairs) if test_pairs else 0
        avg_response_time = total_time / len(test_pairs) if test_pairs else 0

        # Compile evaluation metrics
        evaluation_metrics = {
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "results": results,
        }

        # Save evaluation metrics
        with open(
            self.output_dir / "evaluation_metrics.json", "w", encoding="utf-8"
        ) as f:
            json.dump(evaluation_metrics, f, indent=2)

        return evaluation_metrics


if __name__ == "__main__":
    # Example usage
    fine_tuner = FineTuner(
        model_name="distilgpt2", output_dir="../../models/fine_tuned", use_peft=True
    )

    # Fine-tune the model
    # fine_tuner.fine_tune("../../data/qa_pairs/financial_qa_train.json")

    # Evaluate the model
    # metrics = fine_tuner.evaluate("../../data/qa_pairs/financial_qa_test.json")
    # print(f"Accuracy: {metrics['accuracy']:.2f}")
    # print(f"Average response time: {metrics['avg_response_time']:.3f}s")
