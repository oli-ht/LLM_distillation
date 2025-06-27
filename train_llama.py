#!/usr/bin/env python3
"""
Fine-tuning script for Llama 3.2 1B using LoRA on Serenity QA dataset.
Enhanced with loss monitoring for response distillation.
"""

import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os
from typing import List, Dict
import numpy as np
from torch.nn import functional as F

class ResponseDistillationTrainer(Trainer):
    """
    Custom trainer that shows detailed loss information for response distillation.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute loss with detailed logging for response distillation.
        """
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get labels
        labels = inputs.get("labels")
        
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate task loss (cross-entropy)
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Log detailed loss information
        if self.state.global_step % self.args.logging_steps == 0:
            print(f"\n=== Step {self.state.global_step} Loss Breakdown ===")
            print(f"Task Loss: {task_loss.item():.4f}")
            
            # Calculate perplexity
            perplexity = torch.exp(task_loss).item()
            print(f"Perplexity: {perplexity:.4f}")
            
            # Show some token predictions for debugging
            if shift_logits.size(0) > 0:
                pred_tokens = torch.argmax(shift_logits[0, -10:], dim=-1)
                true_tokens = shift_labels[0, -10:]
                print(f"Last 10 predicted tokens: {pred_tokens.tolist()}")
                print(f"Last 10 true tokens: {true_tokens.tolist()}")
                print(f"Accuracy on last 10: {(pred_tokens == true_tokens).float().mean().item():.2f}")
            
            # Show sample response generation
            if shift_logits.size(0) > 0:
                sample_input = inputs["input_ids"][0, :50]  # First 50 tokens
                print(f"Sample input tokens: {sample_input.tolist()}")
                print("Sample generation disabled to avoid tokenizer issues")
            
            print("=" * 50)
        
        return (task_loss, outputs) if return_outputs else task_loss

def load_model_and_tokenizer(model_name: str, load_8bit: bool = False):
    """
    Load Llama model and tokenizer.
    
    Args:
        model_name (str): Hugging Face model name or local path
        load_8bit (bool): Whether to use 8-bit quantization (disabled on Mac)
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model and tokenizer from: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model on CPU
    model_kwargs = {
        "torch_dtype": torch.float32,  # Use float32 for Mac compatibility
        "trust_remote_code": True,
        "device_map": "cpu",  # Force CPU
    }
    
    print("Forcing model to load on CPU (no GPU/MPS)")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Ensure model is in training mode and parameters require gradients
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def setup_lora_config():
    """
    Configure LoRA parameters for efficient fine-tuning.
    
    Returns:
        LoraConfig: LoRA configuration
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank - higher = more parameters but better performance
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.1,  # Dropout rate
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",  # Attention modules for Llama
            "gate_proj", "up_proj", "down_proj"      # MLP modules for Llama
        ],
        bias="none",  # Don't train bias terms
    )
    
    print("LoRA configuration created:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    print(f"  - Target modules: {lora_config.target_modules}")
    
    return lora_config

def load_qa_dataset(jsonl_file: str) -> List[Dict]:
    """
    Load QA dataset from JSONL file.
    
    Args:
        jsonl_file (str): Path to JSONL file with QA pairs
        
    Returns:
        List[Dict]: List of QA pairs
    """
    print(f"Loading dataset from: {jsonl_file}")
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} QA pairs")
    return data

def prepare_dataset_for_training(data: List[Dict], tokenizer, max_length: int = 512, val_split: float = 0.1):
    """
    Prepare dataset for training by tokenizing and formatting.
    
    Args:
        data (List[Dict]): Raw QA data
        tokenizer: Hugging Face tokenizer
        max_length (int): Maximum sequence length
        val_split (float): Fraction of data to use for validation (0.1 = 10%)
        
    Returns:
        tuple: (train_dataset, val_dataset) - Tokenized datasets ready for training
    """
    print("Preparing dataset for training...")
    
    def format_instruction(instruction: str, input_text: str, output: str) -> str:
        """Format instruction, input, and output into a single text."""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    def tokenize_function(examples):
        """Tokenize the examples."""
        # Format the texts
        texts = []
        for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
            text = format_instruction(instruction, input_text, output)
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set labels to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Convert to Dataset
    dataset = Dataset.from_list(data)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=val_split, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Tokenize both datasets
    train_tokenized = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    print(f"Dataset prepared with max_length={max_length}")
    return train_tokenized, val_tokenized

def setup_training_args(output_dir: str = "./serenity-llama-finetuned"):
    """
    Set up training arguments.
    
    Args:
        output_dir (str): Directory to save the model
        
    Returns:
        TrainingArguments: Training configuration
    """
    # Check if we can use fp16 (CUDA only)
    use_fp16 = torch.cuda.is_available()
    
    # Adjust batch size based on device
    if torch.backends.mps.is_available():
        batch_size = 1  # M3 GPU - smaller batch for 1B model
        print("Using M3 GPU - smaller batch size for Llama 3.2 1B")
    elif torch.cuda.is_available():
        batch_size = 2  # CUDA GPU can handle larger batches
        print("Using CUDA GPU - enabling fp16 and larger batch size")
    else:
        batch_size = 1  # CPU needs smaller batches
        print("Using CPU - smaller batch size")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=batch_size,  # Adjusted batch size
        gradient_accumulation_steps=8,  # Increased for smaller batch size
        learning_rate=2e-4,  # Learning rate
        fp16=use_fp16,  # Enable fp16 only on CUDA
        logging_steps=10,  # Log every 10 steps
        save_steps=100,  # Save checkpoint every 100 steps
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",
        eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy
        load_best_model_at_end=True,
        report_to="wandb",  # Enable Weights & Biases logging
        remove_unused_columns=False,
        warmup_steps=100,  # Warmup steps
        weight_decay=0.01,  # Weight decay
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Enable for memory efficiency with 1B model
    )
    
    print("Training arguments configured:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - FP16 enabled: {use_fp16}")
    
    return training_args

def main():
    """Main training function."""
    
    # Configuration - use Llama 3.2 1B for distillation
    model_name = "meta-llama/Llama-3.2-1B"  # Official Llama 3.2 1B from Hugging Face
    dataset_file = "serenity_instructions.jsonl"
    output_dir = "./serenity-llama-finetuned"
    
    print("=== LLAMA 3.2 1B RESPONSE DISTILLATION WITH LoRA ===")
    print(f"Student Model: {model_name} (1B parameters)")
    print(f"Teacher Knowledge: {dataset_file} (pre-generated QA pairs)")
    print(f"Output: {output_dir}")
    print("This is response distillation - learning from teacher's outputs!")
    print("Using Llama 3.2 1B model for better performance!")
    
    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found!")
        print("Please run prepare_data.py first to create the instruction dataset.")
        return
    
    # Step 1: Load student model and tokenizer
    print("\n1. Loading student model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Step 2: Setup LoRA
    print("\n2. Setting up LoRA...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Ensure the model is properly configured for training
    model.train()
    model.enable_input_require_grads()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Verify that some parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
    
    # Step 3: Load and prepare dataset
    print("\n3. Loading and preparing dataset...")
    raw_data = load_qa_dataset(dataset_file)
    train_dataset, val_dataset = prepare_dataset_for_training(raw_data, tokenizer)
    
    # Step 4: Setup training arguments
    print("\n4. Setting up training arguments...")
    training_args = setup_training_args(output_dir)
    
    # Step 5: Setup data collator
    print("\n5. Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked
    )
    
    # Step 6: Initialize custom trainer
    print("\n6. Initializing trainer...")
    trainer = ResponseDistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        data_collator=data_collator,
    )
    
    # Step 7: Start training
    print("\n7. Starting response distillation training...")
    print("=" * 50)
    print("You'll see detailed loss information every 10 steps!")
    print("The model is learning to mimic your teacher's responses.")
    print("=" * 50)
    trainer.train()
    
    # Step 8: Save the model
    print("\n8. Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n=== RESPONSE DISTILLATION COMPLETE ===")
    print(f"Model saved to: {output_dir}")
    print("Your Llama 3.2 1B model has learned to mimic the teacher's responses!")
    print("You can now use this fine-tuned model with the RAG system!")

if __name__ == "__main__":
    main() 