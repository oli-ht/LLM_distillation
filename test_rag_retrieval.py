#!/usr/bin/env python3
"""
Test script for RAG retrieval system.
This tests the document retrieval part without requiring the fine-tuned model.
"""

import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_documents(json_file):
    """Load email chunks from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_search_index(documents, embedding_model):
    """Create FAISS index for semantic search."""
    print("Creating embeddings for documents...")
    texts = [doc['content'] for doc in documents]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    return index, embeddings

def search_documents(query, index, embedding_model, documents, top_k=3):
    """Search for relevant documents."""
    # Encode the query
    query_embedding = embedding_model.encode([query])
    
    # Search the index
    scores, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Get the relevant documents
    relevant_docs = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(documents):
            doc = documents[idx].copy()
            doc['relevance_score'] = float(score)
            relevant_docs.append(doc)
    
    return relevant_docs

def main():
    """Test the RAG retrieval system."""
    
    # Configuration
    documents_file = "email_chunks.json"
    embedding_model_name = "all-MiniLM-L6-v2"
    
    print("=== RAG RETRIEVAL TEST ===")
    print(f"Loading documents from: {documents_file}")
    
    # Load documents
    documents = load_documents(documents_file)
    print(f"Loaded {len(documents)} document chunks")
    
    # Load embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Create search index
    print("Creating search index...")
    index, embeddings = create_search_index(documents, embedding_model)
    
    # Test questions
    test_questions = [
        "When is the Phase 3A bond check due?",
        "How much is the recreation fee?",
        "What permits are required for the project?",
        "What is the status of the paving work?",
        "How are fees submitted to the county?",
        "What is the pump station status?",
        "When are the AIA documents due?"
    ]
    
    print(f"\n=== TESTING RETRIEVAL ===")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 50)
        
        # Search for relevant documents
        relevant_docs = search_documents(question, index, embedding_model, documents, top_k=3)
        
        # Display results
        for j, doc in enumerate(relevant_docs, 1):
            print(f"\nTop {j} (Score: {doc['relevance_score']:.3f}):")
            print(f"Content: {doc['content'][:200]}...")
            print(f"Length: {len(doc['content'])} characters")
        
        print()

def interactive_search():
    """Interactive search mode."""
    # Configuration
    documents_file = "email_chunks.json"
    embedding_model_name = "all-MiniLM-L6-v2"
    
    print("=== INTERACTIVE RAG SEARCH ===")
    
    # Load everything
    print("Loading documents and model...")
    documents = load_documents(documents_file)
    embedding_model = SentenceTransformer(embedding_model_name)
    index, embeddings = create_search_index(documents, embedding_model)
    
    print(f"Ready! Loaded {len(documents)} document chunks.")
    print("Ask questions about the Serenity project. Type 'quit' to exit.")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not question:
                print("Please enter a question.")
                continue
            
            # Search
            print("Searching...")
            relevant_docs = search_documents(question, index, embedding_model, documents, top_k=3)
            
            # Display results
            print(f"\nFound {len(relevant_docs)} relevant documents:")
            
            for i, doc in enumerate(relevant_docs, 1):
                print(f"\n{i}. (Relevance: {doc['relevance_score']:.3f})")
                print(f"   {doc['content'][:300]}...")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Choose mode
    print("Choose mode:")
    print("1. Run predefined tests")
    print("2. Interactive search")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        interactive_search()
    else:
        print("Invalid choice. Running predefined tests.")
        main() 