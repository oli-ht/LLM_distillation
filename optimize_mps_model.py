#!/usr/bin/env python3
"""
Optimized model loader for Apple M3 with MPS acceleration.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import time

def optimize_for_m3():
    """Optimize PyTorch settings for Apple M3."""
    print("🍎 Optimizing for Apple M3...")
    
    # Enable MPS optimizations
    if torch.backends.mps.is_available():
        # Clear any existing cache
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        print("✅ MPS optimizations enabled")
    else:
        print("⚠️  MPS not available")

def load_optimized_model(checkpoint_path="./serenity-llama-rag-memory-optimized/checkpoint-50"):
    """Load model with M3 optimizations."""
    print(f"📦 Loading optimized model from: {checkpoint_path}")
    
    # Optimize for M3
    optimize_for_m3()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with optimizations
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16,  # Use float16 for better MPS performance
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="mps"  # Directly map to MPS
        )
        
        # Load checkpoint
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        print(f"✅ Optimized model loaded on MPS!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading optimized model: {e}")
        return None, None

def fast_generate(model, tokenizer, question, max_tokens=50):
    """Fast generation with optimizations."""
    print(f"❓ Question: {question}")
    print("⚡ Generating fast response...")
    
    start_time = time.time()
    
    # Create prompt
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    
    # Tokenize with optimizations
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256  # Shorter for speed
    )
    
    # Move to MPS
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with optimizations
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,  # Shorter responses
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache
            repetition_penalty=1.1  # Prevent repetition
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    generation_time = time.time() - start_time
    
    print(f"🤖 Response: {response}")
    print(f"⏱️  Generation time: {generation_time:.2f}s")
    
    return response, generation_time

def benchmark_model(model, tokenizer):
    """Benchmark the model performance."""
    print("\n🏃‍♂️ BENCHMARKING MODEL PERFORMANCE")
    print("=" * 50)
    
    test_questions = [
        "What ensures traffic control compliance?",
        "What is Serenity?",
        "How many lots are in the development?"
    ]
    
    total_time = 0
    responses = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/3 ---")
        response, gen_time = fast_generate(model, tokenizer, question, max_tokens=30)
        total_time += gen_time
        responses.append(response)
    
    avg_time = total_time / len(test_questions)
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"  - Average generation time: {avg_time:.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Speed: {1/avg_time:.1f} responses/second")
    
    if avg_time < 5:
        print("🚀 Excellent performance!")
    elif avg_time < 10:
        print("✅ Good performance")
    else:
        print("⚠️  Slow performance - consider optimizations")

def interactive_fast_mode(model, tokenizer):
    """Interactive mode with fast generation."""
    if model is None or tokenizer is None:
        print("❌ Model not loaded properly")
        return
    
    print("\n🎮 FAST INTERACTIVE MODE - M3 Optimized")
    print("Ask questions for quick responses!")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Fast generation
            response, gen_time = fast_generate(model, tokenizer, question, max_tokens=50)
            
            if gen_time > 10:
                print("💡 Tip: Try shorter questions for faster responses")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🍎 M3 OPTIMIZED MODEL LOADER")
    print("=" * 50)
    
    # Load optimized model
    model, tokenizer = load_optimized_model()
    
    if model is None:
        print("❌ Failed to load optimized model")
        return
    
    # Check command line arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            benchmark_model(model, tokenizer)
        elif sys.argv[1] == "interactive":
            interactive_fast_mode(model, tokenizer)
        else:
            # Single test
            fast_generate(model, tokenizer, "What ensures traffic control compliance?")
    else:
        # Run benchmark by default
        benchmark_model(model, tokenizer)
        print("\n💡 Run 'python optimize_mps_model.py interactive' for fast interactive mode")

if __name__ == "__main__":
    main() 