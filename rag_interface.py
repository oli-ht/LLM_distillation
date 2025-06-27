#!/usr/bin/env python3
"""
Simple interface for the RAG system.
Allows interactive testing of the question-answering system.
"""

import json
from rag_system import RAGSystem

def load_rag_system():
    """Load the RAG system with proper error handling."""
    try:
        # Configuration - update these paths as needed
        model_path = "./serenity-llama-finetuned"
        documents_file = "email_chunks.json"
        
        print("Initializing RAG system...")
        rag = RAGSystem(model_path, documents_file)
        return rag
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files: {e}")
        print("Please ensure:")
        print("1. Fine-tuned model exists at: ./serenity-llama-finetuned")
        print("2. Email chunks file exists at: email_chunks.json")
        print("3. Run prepare_emails.py first to create email_chunks.json")
        return None
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None

def interactive_mode(rag):
    """Run interactive question-answering mode."""
    print("\n" + "="*60)
    print("SERENITY PROJECT RAG SYSTEM")
    print("="*60)
    print("Ask questions about the Serenity development project.")
    print("The system will search through email documents to find relevant information.")
    print("Type 'quit' to exit.")
    print("="*60)
    
    while True:
        try:
            # Get user question
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not question:
                print("Please enter a question.")
                continue
            
            # Get answer
            print("\nProcessing your question...")
            response = rag.answer_question(question)
            
            # Display answer
            print(f"\nAnswer: {response['answer']}")
            
            # Show metadata
            print(f"\nSources used: {response['context_used']} document(s)")
            
            # Show top source
            if response['relevant_documents']:
                top_doc = response['relevant_documents'][0]
                print(f"Top source (relevance: {top_doc['relevance_score']:.3f}):")
                print(f"  {top_doc['content'][:150]}...")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing question: {e}")

def demo_mode(rag):
    """Run demo with predefined questions."""
    demo_questions = [
        "When is the Phase 3A bond check due?",
        "How much is the recreation fee and how is it calculated?",
        "What permits are required for the project?",
        "What is the status of the paving work?",
        "How are fees submitted to the county?"
    ]
    
    print("\n" + "="*60)
    print("DEMO MODE - Testing with predefined questions")
    print("="*60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 40)
        
        try:
            response = rag.answer_question(question)
            print(f"Answer: {response['answer']}")
            print(f"Sources: {response['context_used']} document(s)")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

def main():
    """Main function."""
    print("Loading RAG system...")
    rag = load_rag_system()
    
    if rag is None:
        print("Failed to load RAG system. Please check the configuration.")
        return
    
    # Choose mode
    print("\nChoose mode:")
    print("1. Interactive mode (ask your own questions)")
    print("2. Demo mode (test with predefined questions)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_mode(rag)
            break
        elif choice == "2":
            demo_mode(rag)
            break
        else:
            print("Please enter 1 or 2.")

if __name__ == "__main__":
    main() 