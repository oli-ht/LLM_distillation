#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) system for Serenity project Q&A.
This system retrieves relevant email content and uses it to answer questions.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import faiss
import pickle

class RAGSystem:
    def __init__(self, model_path: str, documents_file: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.
        
        Args:
            model_path (str): Path to fine-tuned Llama model
            documents_file (str): Path to processed email chunks JSON file
            embedding_model (str): Sentence transformer model for embeddings
        """
        self.model_path = model_path
        self.documents_file = documents_file
        
        print("Loading fine-tuned model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        print("Loading documents...")
        self.documents = self.load_documents()
        
        print("Creating search index...")
        self.index, self.document_embeddings = self.create_search_index()
        
        print("RAG system initialized successfully!")
    
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
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query (str): User's question
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: List of relevant documents
        """
        # Encode the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get the relevant documents
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def create_context_prompt(self, question: str, relevant_docs: List[Dict]) -> str:
        """
        Create a prompt with retrieved context for the model.
        
        Args:
            question (str): User's question
            relevant_docs (List[Dict]): Retrieved relevant documents
            
        Returns:
            str: Formatted prompt with context
        """
        # Combine relevant documents into context
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""Based on the following email documents about the Serenity development project, answer the question below.

Context from emails:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate an answer using the fine-tuned model.
        
        Args:
            prompt (str): Input prompt with context
            max_length (int): Maximum length of generated response
            
        Returns:
            str: Generated answer
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question (str): User's question
            top_k (int): Number of documents to retrieve
            
        Returns:
            Dict: Answer with metadata
        """
        print(f"Processing question: {question}")
        
        # Step 1: Retrieve relevant documents
        print("Retrieving relevant documents...")
        relevant_docs = self.retrieve_relevant_documents(question, top_k)
        
        # Step 2: Create context prompt
        print("Creating context prompt...")
        prompt = self.create_context_prompt(question, relevant_docs)
        
        # Step 3: Generate answer
        print("Generating answer...")
        answer = self.generate_answer(prompt)
        
        # Step 4: Prepare response
        response = {
            "question": question,
            "answer": answer,
            "relevant_documents": relevant_docs,
            "context_used": len(relevant_docs)
        }
        
        return response

def main():
    """Example usage of the RAG system."""
    
    # Configuration
    model_path = "./serenity-llama-finetuned"  # Path to your fine-tuned model
    documents_file = "email_chunks.json"
    
    # Initialize RAG system
    rag = RAGSystem(model_path, documents_file)
    
    # Example questions
    test_questions = [
        "When is the Phase 3A bond check due?",
        "How much is the recreation fee?",
        "What permits are required for the project?"
    ]
    
    print("\n=== TESTING RAG SYSTEM ===")
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        
        try:
            response = rag.answer_question(question)
            
            print(f"Answer: {response['answer']}")
            print(f"Documents used: {response['context_used']}")
            
            # Show top document
            if response['relevant_documents']:
                top_doc = response['relevant_documents'][0]
                print(f"Top document score: {top_doc['relevance_score']:.3f}")
                print(f"Top document preview: {top_doc['content'][:100]}...")
                
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main() 