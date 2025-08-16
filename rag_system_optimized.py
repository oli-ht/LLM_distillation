#!/usr/bin/env python3
"""
OPTIMIZED RAG (Retrieval-Augmented Generation) system for Serenity project Q&A.
This system addresses the slow generation and accuracy issues in the original implementation.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import faiss
import time

class OptimizedRAGSystem:
    def __init__(self, model_path: str, documents_file: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the OPTIMIZED RAG system.
        
        Args:
            model_path (str): Path to fine-tuned Llama model
            documents_file (str): Path to processed email chunks JSON file
            embedding_model (str): Sentence transformer model for embeddings
        """
        self.model_path = model_path
        self.documents_file = documents_file
        
        print("🚀 Loading OPTIMIZED fine-tuned model...")
        
        # Use 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",  # Let transformers handle device placement
            low_cpu_mem_usage=True
        )
        
        print("📚 Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        print("📄 Loading documents...")
        self.documents = self.load_documents()
        
        print("🔍 Creating search index...")
        self.index, self.document_embeddings = self.create_search_index()
        
        print("✅ OPTIMIZED RAG system initialized successfully!")
    
    def load_documents(self) -> List[Dict]:
        """Load processed email documents."""
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_search_index(self) -> Tuple[faiss.Index, np.ndarray]:
        """Create FAISS index for semantic search."""
        # Create embeddings for all documents
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        return index, embeddings
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant documents for a query.
        Increased top_k for better context coverage.
        """
        # Encode the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get the relevant documents with better filtering
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents) and score > 0.1:  # Filter low relevance
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def create_context_prompt(self, question: str, relevant_docs: List[Dict]) -> str:
        """
        Create an OPTIMIZED prompt with retrieved context for the model.
        """
        if not relevant_docs:
            return f"Question: {question}\n\nAnswer:"
        
        # Create a more focused context
        context_parts = []
        for i, doc in enumerate(relevant_docs[:3]):  # Limit to top 3 for efficiency
            # Truncate long documents to avoid token limit issues
            content = doc['content'][:500] if len(doc['content']) > 500 else doc['content']
            context_parts.append(f"Document {i+1}: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Create a clear, focused prompt
        prompt = f"""Based on the following information, answer the question concisely and accurately:

{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Generate an answer using OPTIMIZED generation parameters.
        Reduced max_new_tokens for faster generation.
        """
        try:
            # Tokenize with reasonable limits
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024  # Reduced from 2048
            )
            
            # Move inputs to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # OPTIMIZED generation parameters for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.3,  # Reduced for more focused responses
                    do_sample=True,
                    top_p=0.9,  # Add nucleus sampling
                    repetition_penalty=1.1,  # Prevent repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,  # Stop when EOS is generated
                    use_cache=True  # Enable KV cache for speed
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict:
        """
        Answer a question using OPTIMIZED RAG.
        """
        start_time = time.time()
        
        print(f"🔍 Processing question: {question}")
        
        # Step 1: Retrieve relevant documents
        print("📚 Retrieving relevant documents...")
        relevant_docs = self.retrieve_relevant_documents(question, top_k)
        
        # Step 2: Create context prompt
        print("📝 Creating context prompt...")
        prompt = self.create_context_prompt(question, relevant_docs)
        
        # Step 3: Generate answer with timing
        print("⚡ Generating answer...")
        generation_start = time.time()
        answer = self.generate_answer(prompt)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Step 4: Prepare response
        response = {
            "question": question,
            "answer": answer,
            "relevant_documents": relevant_docs,
            "context_used": len(relevant_docs),
            "generation_time": generation_time,
            "total_time": total_time,
            "prompt_tokens": len(self.tokenizer.encode(prompt)),
            "answer_tokens": len(self.tokenizer.encode(answer))
        }
        
        print(f"✅ Generated in {generation_time:.2f}s (Total: {total_time:.2f}s)")
        
        return response

def main():
    """Example usage of the OPTIMIZED RAG system."""
    
    # Configuration
    model_path = "./serenity-llama-finetuned"  # Path to your fine-tuned model
    documents_file = "email_chunks.json"
    
    # Initialize OPTIMIZED RAG system
    rag = OptimizedRAGSystem(model_path, documents_file)
    
    # Example questions
    test_questions = [
        "When is the Phase 3A bond check due?",
        "How much is the recreation fee?",
        "What permits are required for the project?",
        "Phase 4B status and requirements"
    ]
    
    print("\n=== TESTING OPTIMIZED RAG SYSTEM ===")
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        
        try:
            response = rag.answer_question(question)
            
            print(f"Answer: {response['answer']}")
            print(f"Documents used: {response['context_used']}")
            print(f"Generation time: {response['generation_time']:.2f}s")
            print(f"Total time: {response['total_time']:.2f}s")
            print(f"Tokens: {response['prompt_tokens']} + {response['answer_tokens']}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 