#!/usr/bin/env python3
"""
Test script specifically for the serenity-llama-rag-memory-optimized model.
"""

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

class RAGModelTester:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Serenity-specific test questions
        self.test_questions = [
            "What is the Serenity project about?",
            "What technologies are used in Serenity?",
            "How does the team communicate in Serenity?",
            "What is the development workflow in Serenity?",
            "What are the main features of Serenity?",
            "Who are the key team members in Serenity?",
            "What are the current challenges in Serenity?",
            "How is Serenity different from other projects?",
            "What is the timeline for Serenity development?",
            "What are the success metrics for Serenity?"
        ]
    
    def load_model(self, model_path="./serenity-llama-rag-memory-optimized"):
        """Load the RAG memory-optimized model."""
        print(f"📦 Loading RAG model from: {model_path}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load fine-tuned model
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ RAG model loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    
    def generate_response(self, model, tokenizer, question, max_length=512):
        """Generate a response for a given question."""
        try:
            # Format the prompt
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
    
    def test_model(self):
        """Test the RAG model with all questions."""
        print("🧪 TESTING SERENITY RAG MEMORY-OPTIMIZED MODEL")
        print("=" * 80)
        
        # Load model
        model, tokenizer = self.load_model()
        if model is None:
            print("❌ Could not load model. Make sure the model exists and you have access to Llama 3.2 1B.")
            return
        
        results = []
        
        # Test each question
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            
            # Generate response
            start_time = time.time()
            response = self.generate_response(model, tokenizer, question)
            generation_time = time.time() - start_time
            
            # Store results
            result = {
                "question": question,
                "response": response,
                "generation_time": generation_time
            }
            results.append(result)
            
            # Print results
            print(f"💬 Response ({generation_time:.2f}s):")
            print(f"   {response}")
            print("-" * 80)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """Save test results to file."""
        output_file = "rag_model_test_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {output_file}")
    
    def interactive_mode(self):
        """Interactive testing mode."""
        print(f"\n🎮 INTERACTIVE TESTING MODE")
        print("Ask any Serenity-related questions!")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        # Load model
        model, tokenizer = self.load_model()
        if model is None:
            return
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("🤔 Generating response...")
            start_time = time.time()
            response = self.generate_response(model, tokenizer, question)
            generation_time = time.time() - start_time
            
            print(f"💬 Response ({generation_time:.2f}s):")
            print(f"   {response}")
            print("-" * 50)

def main():
    """Main testing function."""
    tester = RAGModelTester()
    
    print("🧪 SERENITY RAG MODEL TESTER")
    print("=" * 80)
    
    # Check if model exists
    model_path = "./serenity-llama-rag-memory-optimized"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run training first to create the model.")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Check if user wants interactive mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        tester.interactive_mode()
    else:
        # Run standard testing
        results = tester.test_model()
        
        if results:
            print(f"\n🎯 Testing complete! Tested {len(results)} questions.")
            print("Run 'python test_rag_model.py interactive' for interactive testing.")

if __name__ == "__main__":
    main() 