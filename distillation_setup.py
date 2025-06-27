#!/usr/bin/env python3
"""
Setup script for teacher models in LLM distillation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def setup_teacher_model(teacher_type="llama_8b", model_path=None):
    """
    Setup different types of teacher models for distillation.
    
    Args:
        teacher_type (str): Type of teacher model
        model_path (str): Custom path to teacher model
        
    Returns:
        tuple: (teacher_model, teacher_tokenizer)
    """
    
    print(f"Setting up teacher model: {teacher_type}")
    
    if teacher_type == "llama_8b":
        # Use Llama 3.2 8B as teacher
        teacher_name = "meta-llama/Llama-3.2-8B"
        print(f"Loading Llama 3.2 8B teacher from: {teacher_name}")
        
    elif teacher_type == "llama_70b":
        # Use Llama 3.2 70B as teacher (requires more GPU memory)
        teacher_name = "meta-llama/Llama-3.2-70B"
        print(f"Loading Llama 3.2 70B teacher from: {teacher_name}")
        
    elif teacher_type == "gpt2_large":
        # Use GPT-2 Large as teacher (smaller, easier to load)
        teacher_name = "gpt2-large"
        print(f"Loading GPT-2 Large teacher from: {teacher_name}")
        
    elif teacher_type == "custom":
        # Use custom teacher model
        if not model_path:
            raise ValueError("model_path must be provided for custom teacher")
        teacher_name = model_path
        print(f"Loading custom teacher from: {teacher_name}")
        
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")
    
    # Load teacher tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    
    # Load teacher model with optimizations
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set teacher to evaluation mode
    teacher_model.eval()
    
    # Freeze teacher parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    print(f"Teacher model loaded successfully!")
    print(f"Teacher model size: {teacher_model.num_parameters():,} parameters")
    
    return teacher_model, teacher_tokenizer

def test_teacher_model(teacher_model, teacher_tokenizer, test_prompt="What is machine learning?"):
    """
    Test the teacher model with a sample prompt.
    """
    print(f"\nTesting teacher model with prompt: '{test_prompt}'")
    
    # Tokenize input
    inputs = teacher_tokenizer(test_prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = teacher_model.generate(
            inputs["input_ids"],
            max_length=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=teacher_tokenizer.eos_token_id
        )
    
    # Decode response
    response = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Teacher response: {response}")
    
    return response

def main():
    """Main function to demonstrate teacher model setup."""
    
    print("=== TEACHER MODEL SETUP FOR DISTILLATION ===")
    
    # Choose your teacher model type
    teacher_options = {
        "1": ("llama_8b", "Llama 3.2 8B (recommended)"),
        "2": ("llama_70b", "Llama 3.2 70B (requires more GPU memory)"),
        "3": ("gpt2_large", "GPT-2 Large (smaller, easier to load)"),
        "4": ("custom", "Custom model path")
    }
    
    print("\nAvailable teacher models:")
    for key, (model_type, description) in teacher_options.items():
        print(f"  {key}. {description}")
    
    choice = input("\nSelect teacher model (1-4): ").strip()
    
    if choice not in teacher_options:
        print("Invalid choice. Using Llama 3.2 8B as default.")
        choice = "1"
    
    teacher_type = teacher_options[choice][0]
    
    # Get custom path if needed
    model_path = None
    if teacher_type == "custom":
        model_path = input("Enter path to your custom teacher model: ").strip()
        if not os.path.exists(model_path):
            print(f"Error: Model path '{model_path}' does not exist!")
            return
    
    try:
        # Setup teacher model
        teacher_model, teacher_tokenizer = setup_teacher_model(teacher_type, model_path)
        
        # Test the teacher
        test_teacher_model(teacher_model, teacher_tokenizer)
        
        print(f"\n=== TEACHER MODEL READY ===")
        print(f"Teacher type: {teacher_type}")
        if model_path:
            print(f"Teacher path: {model_path}")
        
        # Show how to use in training script
        print(f"\nTo use this teacher in your training script:")
        print(f"1. Set use_distillation = True")
        print(f"2. Set teacher_model_name = '{teacher_type if teacher_type != 'custom' else model_path}'")
        print(f"3. Adjust temperature and alpha as needed")
        
    except Exception as e:
        print(f"Error setting up teacher model: {e}")
        print("Make sure you have access to the model and sufficient GPU memory.")

if __name__ == "__main__":
    main() 