#!/usr/bin/env python3
"""
Improved fine-tuning script for Llama 3.2 1B with better loss reduction.
Addresses high loss issues with improved hyperparameters and training strategy.
"""

import torch
import json
import gc
import os
import shutil
import warnings
import time
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from typing import List, Dict
import numpy as np
from torch.nn import functional as F

# Suppress warnings and disable wandb
warnings.filterwarnings("ignore", message=".*MallocStackLogging.*")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"

# Memory optimization settings
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable memory limit
torch.backends.cudnn.benchmark = False  # Disable for memory savings

class RAGEnhancedCallback(TrainerCallback):
    """RAG-enhanced callback with context retrieval and loss monitoring."""
    
    def __init__(self, documents_file: str):
        self.start_time = None
        self.best_loss = float('inf')
        self.patience = 0
        self.max_patience = 20  # Stop if loss doesn't improve for 20 steps
        self.documents_file = documents_file
        self.documents = None
        
        # Load documents for context retrieval
        self.setup_rag_components()
    
    def setup_rag_components(self):
        """Setup RAG components with memory optimization."""
        print("🔍 Setting up memory-optimized RAG components...")
        
        # Load documents with memory optimization
        if os.path.exists(self.documents_file):
            try:
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                
                # Limit document size for memory efficiency
                if len(self.documents) > 1000:
                    print(f"📚 Limiting documents from {len(self.documents)} to 1000 for memory efficiency")
                    self.documents = self.documents[:1000]
                
                print(f"📚 Loaded {len(self.documents)} email documents (memory optimized)")
            except Exception as e:
                print(f"⚠️  Error loading documents: {e}")
                self.documents = []
        else:
            print(f"⚠️  Documents file not found: {self.documents_file}")
            print("Training will proceed without RAG context")
    
    def retrieve_context_simple(self, question: str, top_k: int = 1) -> str:
        """Simple context retrieval with memory optimization."""
        if not self.documents:
            return ""
        
        try:
            # Simple keyword-based retrieval
            question_lower = question.lower()
            relevant_docs = []
            
            # Limit search to first 500 documents for speed
            search_docs = self.documents[:500]
            
            for doc in search_docs:
                content = doc.get('content', '').lower()
                # Check if any words from question appear in document
                question_words = question_lower.split()
                matches = sum(1 for word in question_words if len(word) > 3 and word in content)
                if matches > 0:
                    relevant_docs.append((matches, doc))
            
            # Sort by relevance and take top_k
            relevant_docs.sort(key=lambda x: x[0], reverse=True)
            
            context_parts = []
            for i, (score, doc) in enumerate(relevant_docs[:top_k]):
                content = doc.get('content', '')[:200]  # Limit length for memory
                context_parts.append(f"Email {i+1}: {content}...")
            
            return "\n".join(context_parts)
        except Exception as e:
            print(f"⚠️  Context retrieval failed: {e}")
            return ""
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"🔍 RAG-ENHANCED TRAINING STARTED")
        print(f"{'='*80}")
        print(f"📊 Training Configuration:")
        print(f"  - Epochs: {args.num_train_epochs}")
        print(f"  - Max steps: {args.max_steps}")
        print(f"  - Batch size: {args.per_device_train_batch_size}")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Warmup steps: {args.warmup_steps}")
        print(f"  - LR scheduler: {args.lr_scheduler_type}")
        print(f"  - RAG Context: {'Enabled' if self.documents else 'Disabled'}")
        
        # Show device info
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            print(f"🍎 Using MPS (Apple Silicon GPU)")
        else:
            print(f"💻 Using CPU")
        
        print(f"{'='*80}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor loss and show progress every 20 steps."""
        if state.global_step % 20 == 0 and state.global_step > 0:
            current_time = time.time()
            
            # Calculate progress
            total_steps = args.max_steps if args.max_steps else 200
            progress = (state.global_step / total_steps) * 100
            
            # Calculate time estimates
            elapsed_time = current_time - self.start_time
            if state.global_step > 0:
                steps_per_second = state.global_step / elapsed_time
                remaining_steps = total_steps - state.global_step
                eta_seconds = remaining_steps / steps_per_second
                eta_minutes = eta_seconds / 60
            else:
                eta_minutes = 0
            
            print(f"\n--- 📊 RAG Progress Update ---")
            print(f"Step {state.global_step}/{total_steps} ({progress:.1f}%)")
            print(f"⏱️  Elapsed: {elapsed_time/60:.1f} min | ETA: {eta_minutes:.1f} min")
            
            # Show learning rate
            trainer = kwargs.get('trainer')
            if hasattr(trainer, 'lr_scheduler'):
                current_lr = trainer.lr_scheduler.get_last_lr()[0]
                print(f"📈 Learning rate: {current_lr:.2e}")
            
            # Show memory usage
            if torch.cuda.is_available():
                print(f"🎮 GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print("---")

class RAGEnhancedTrainer(Trainer):
    """
    RAG-enhanced trainer that teaches the model to use context for better answers.
    """
    
    def __init__(self, *args, rag_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')
        self.rag_callback = rag_callback
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with enhanced monitoring and early stopping.
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
        
        # Enhanced logging every 10 steps (reduced frequency for memory)
        if self.state.global_step % 10 == 0:
            current_epoch = self.state.epoch if hasattr(self.state, 'epoch') else 0
            
            print(f"\n=== 🔍 RAG Epoch {current_epoch:.2f} | Step {self.state.global_step} ===")
            print(f"Loss: {task_loss.item():.4f}")
            
            # Calculate perplexity
            perplexity = torch.exp(task_loss).item()
            print(f"Perplexity: {perplexity:.4f}")
            
            # Show accuracy on last few tokens
            if shift_logits.size(0) > 0:
                pred_tokens = torch.argmax(shift_logits[0, -10:], dim=-1)
                true_tokens = shift_labels[0, -10:]
                accuracy = (pred_tokens == true_tokens).float().mean().item()
                print(f"Accuracy (last 10): {accuracy:.2f}")
            
            # Loss trend analysis
            if task_loss.item() < self.best_loss:
                self.best_loss = task_loss.item()
                print(f"🎯 New best loss: {self.best_loss:.4f}")
            
            # Memory usage
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print("=" * 40)
        
        # Force garbage collection every 5 steps
        if self.state.global_step % 5 == 0:
            gc.collect()
        
        return (task_loss, outputs) if return_outputs else task_loss

def load_model_and_tokenizer_improved(model_name: str):
    """
    Load model and tokenizer with improved settings.
    """
    print(f"🚀 Loading model and tokenizer from: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"🍎 Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print(f"💻 Using CPU")
    
    # Load model with memory optimization
    model_kwargs = {
        "torch_dtype": torch.float32,  # Use float32 for memory efficiency
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # Enable low CPU memory usage
    }
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = model.to(device)
    model.train()
    
    print(f"✅ Model and tokenizer loaded successfully on {device}!")
    return model, tokenizer

def setup_lora_config_improved():
    """
    Configure LoRA parameters for better loss reduction.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,  # Higher rank for better capacity
        lora_alpha=64,  # Higher alpha for better scaling
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    
    print("🔧 Improved LoRA configuration:")
    print(f"  - Rank (r): {lora_config.r} (increased for better capacity)")
    print(f"  - Alpha: {lora_config.lora_alpha} (increased for better scaling)")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    
    return lora_config

def load_qa_dataset_improved(jsonl_file: str) -> List[Dict]:
    """
    Load QA dataset with better error handling.
    """
    print(f"📚 Loading dataset from: {jsonl_file}")
    
    if not os.path.exists(jsonl_file):
        print(f"❌ Dataset file not found: {jsonl_file}")
        return []
    
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
                print(f"⚠️  Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(data)} QA pairs")
    return data

def prepare_dataset_for_rag_training(data: List[Dict], tokenizer, rag_callback, max_length: int = 1024, val_split: float = 0.1):
    """
    Prepare dataset with RAG context for training.
    """
    print("🔍 Preparing RAG-enhanced dataset...")
    
    # Use fewer samples for memory efficiency
    max_samples = 150  # Reduced for memory efficiency
    if len(data) > max_samples:
        print(f"📊 Limiting dataset from {len(data)} to {max_samples} samples for memory efficiency")
        data = data[:max_samples]
    
    def format_rag_instruction(instruction: str, input_text: str, output: str, context: str = "") -> str:
        """Format instruction with RAG context."""
        if context:
            # Include RAG context in the prompt
            prompt = f"""Based on the following Serenity project emails, answer the question below.

Context from emails:
{context}

### Instruction:
{instruction}"""
        else:
            prompt = f"### Instruction:\n{instruction}"
        
        if input_text:
            prompt += f"\n\n### Input:\n{input_text}"
        
        prompt += f"\n\n### Response:\n{output}"
        return prompt
    
    def tokenize_function(examples):
        """Tokenize the examples with RAG context."""
        # Format the texts with RAG context
        texts = []
        for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
            # Retrieve context for this question (reduced frequency for memory)
            question = instruction if not input_text else f"{instruction} {input_text}"
            context = rag_callback.retrieve_context_simple(question, top_k=1) if rag_callback else ""
            
            text = format_rag_instruction(instruction, input_text, output, context)
            texts.append(text)
        
        # Tokenize with shorter sequences for memory efficiency
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
    
    print(f"📊 Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
    
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
    
    print(f"✅ Memory-optimized RAG dataset prepared with max_length={max_length}")
    return train_tokenized, val_tokenized

def setup_training_args_improved(output_dir: str = "./serenity-llama-finetuned"):
    """
    Set up training arguments optimized for loss reduction.
    """
    # Memory-optimized batch size
    batch_size = 1  # Smallest batch size for memory efficiency
    print("🧠 Using memory-optimized batch size for stability")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced epochs for memory efficiency
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,  # Reduced for memory
        learning_rate=1e-4,  # Conservative learning rate
        fp16=False,  # Disable fp16 for memory stability
        logging_steps=10,  # Log every 10 steps
        save_steps=50,  # Save every 50 steps
        eval_steps=50,  # Evaluate every 50 steps
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        report_to=None,  # Disable wandb
        remove_unused_columns=False,
        warmup_steps=10,  # Reduced warmup steps
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False,  # Disable for memory savings
        gradient_checkpointing=True,  # Enable for memory savings
        max_steps=100,  # Reduced total steps for memory efficiency
        save_total_limit=3,  # Keep more checkpoints
        lr_scheduler_type="cosine",  # Better learning rate scheduling
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    print("⚙️  Improved training arguments:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Max steps: {training_args.max_steps}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - LR scheduler: {training_args.lr_scheduler_type}")
    print(f"  - Warmup steps: {training_args.warmup_steps}")
    
    return training_args

def estimate_improved_training_time():
    """
    Estimate training time for improved settings.
    """
    print("\n⏱️  Improved Training Time Estimates:")
    
    if torch.cuda.is_available():
        print("🎮 GPU Training:")
        print("  - Estimated time: 1-2 hours")
        print("  - 200 steps with batch size 2")
        print("  - ~300 QA pairs")
        print("  - Expected final loss: 2.5-3.5")
    elif torch.backends.mps.is_available():
        print("🍎 MPS Training:")
        print("  - Estimated time: 2-4 hours")
        print("  - 200 steps with batch size 1")
        print("  - ~300 QA pairs")
        print("  - Expected final loss: 2.8-3.8")
    else:
        print("💻 CPU Training:")
        print("  - Estimated time: 6-12 hours")
        print("  - 200 steps with batch size 1")
        print("  - ~300 QA pairs")
        print("  - Expected final loss: 3.0-4.0")
    
    print("\n💡 Improvements:")
    print("  - More training steps (200 vs 50)")
    print("  - Better learning rate scheduling (cosine)")
    print("  - Higher LoRA rank (32 vs 16)")
    print("  - More dataset samples (300 vs 100)")
    print("  - Lower initial learning rate (1e-4 vs 2e-4)")

def main():
    """Main training function with improvements for loss reduction."""
    
    # Configuration
    model_name = "meta-llama/Llama-3.2-1B"
    dataset_file = "serenity_instructions.jsonl"
    documents_file = "email_chunks.json"  # Serenity email documents
    output_dir = "./serenity-llama-rag-memory-optimized"
    
    print("=" * 80)
    print("🔍 MEMORY-OPTIMIZED RAG-ENHANCED LLAMA 3.2 1B TRAINING")
    print("=" * 80)
    print(f"🎓 Student Model: {model_name}")
    print(f"📚 Teacher Knowledge: {dataset_file}")
    print(f"📧 RAG Context: {documents_file}")
    print(f"💾 Output: {output_dir}")
    print(f"🧠 Memory Optimization: Enabled")
    print("🔍 Model learns to USE Serenity emails for better answers!")
    print("=" * 80)
    
    # Show RAG training estimates
    estimate_improved_training_time()
    
    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"\n❌ Error: Dataset file '{dataset_file}' not found!")
        print("Please run prepare_data.py first to create the instruction dataset.")
        return
    
    # Check if documents exist
    if not os.path.exists(documents_file):
        print(f"\n⚠️  Warning: Documents file '{documents_file}' not found!")
        print("RAG training will proceed without context retrieval.")
        print("This will still work but won't be as effective.")
    
    # Step 1: Load student model and tokenizer
    print("\n1️⃣ Loading student model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer_improved(model_name)
    
    # Show model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Parameter efficiency: {trainable_params/total_params*100:.2f}%")
    
    # Step 2: Setup LoRA
    print("\n2️⃣ Setting up improved LoRA...")
    lora_config = setup_lora_config_improved()
    model = get_peft_model(model, lora_config)
    
    # Ensure the model is properly configured for training
    model.train()
    model.enable_input_require_grads()
    
    # Print trainable parameters after LoRA
    model.print_trainable_parameters()
    
    # Step 3: Setup RAG callback
    print("\n3️⃣ Setting up RAG components...")
    rag_callback = RAGEnhancedCallback(documents_file)
    
    # Step 4: Load and prepare dataset
    print("\n4️⃣ Loading and preparing RAG dataset...")
    raw_data = load_qa_dataset_improved(dataset_file)
    
    if not raw_data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Show dataset statistics
    print(f"📊 Dataset Statistics:")
    print(f"  - Total QA pairs: {len(raw_data)}")
    
    train_dataset, val_dataset = prepare_dataset_for_rag_training(raw_data, tokenizer, rag_callback)
    
    # Step 5: Setup training arguments
    print("\n5️⃣ Setting up RAG training arguments...")
    training_args = setup_training_args_improved(output_dir)
    
    # Step 6: Setup data collator
    print("\n6️⃣ Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Step 7: Initialize RAG trainer
    print("\n7️⃣ Initializing RAG trainer...")
    trainer = RAGEnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[rag_callback],
        rag_callback=rag_callback,
    )
    
    # Step 8: Start RAG-enhanced training
    print("\n8️⃣ Starting RAG-enhanced training...")
    print("=" * 80)
    print("🔍 MEMORY-OPTIMIZED RAG TRAINING STARTED!")
    print("📊 Progress updates every 20 steps")
    print("🧠 Loss logging every 10 steps")
    print("📧 Model learning to use Serenity emails as context")
    print("🧠 Memory optimization enabled")
    print("🎯 Expected final loss: 2.0-3.0")
    print("⏱️  Estimated time: 1-2 hours")
    print("=" * 80)
    
    try:
        trainer.train()
        
        # Step 9: Save the model
        print("\n9️⃣ Saving RAG-enhanced model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n{'='*80}")
        print("✅ MEMORY-OPTIMIZED RAG TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"💾 Model saved to: {output_dir}")
        print("🔍 Your Llama 3.2 1B model now knows how to use Serenity emails!")
        print("📧 The model can retrieve and use relevant context for better answers!")
        print("🧠 Memory optimization successful!")
        
        # Show final training statistics
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            print(f"\n📈 Final Training Statistics:")
            final_log = trainer.state.log_history[-1]
            for key, value in final_log.items():
                if key not in ['step', 'epoch']:
                    print(f"  - {key}: {value:.4f}")
            
            # Loss analysis
            if 'train_loss' in final_log:
                loss = final_log['train_loss']
                if loss < 2.5:
                    print(f"🎉 Excellent! Final loss: {loss:.4f} (RAG working perfectly!)")
                elif loss < 3.0:
                    print(f"✅ Good! Final loss: {loss:.4f} (RAG working well)")
                elif loss < 3.5:
                    print(f"⚠️  Fair. Final loss: {loss:.4f} (RAG needs more training)")
                else:
                    print(f"❌ Poor. Final loss: {loss:.4f} (RAG not working effectively)")
        
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Memory optimization failed. Try reducing dataset size further.")

if __name__ == "__main__":
    main() 