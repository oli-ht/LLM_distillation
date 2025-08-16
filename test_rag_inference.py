from rag_system_optimized import OptimizedRAGSystem

if __name__ == "__main__":
    # Update these paths if your model or document files are elsewhere
    model_path = "./serenity-llama-rag-memory-optimized"
    documents_file = "email_chunks.json"

    rag = OptimizedRAGSystem(model_path, documents_file)

    print("Type your question and press Enter (Ctrl+C to exit):")
    while True:
        try:
            question = input("\nYour question: ")
            response = rag.answer_question(question)
            print(f"\nModel answer: {response['answer']}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")