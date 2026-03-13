"""
RAG (Retrieval Augmented Generation) pipeline from scratch.

Usage:
  # Step 1 — process the PDF and create embeddings (run once):
  python text_processing.py

  # Step 2 — run the interactive Q&A assistant:
  python main.py
"""

from generate import ask
from semantic_search import print_top_results_and_scores, embeddings


def main():
    print("=" * 60)
    print("Research Paper Q&A Assistant (RAG)")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        query = input("\nAsk a question about the paper: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Goodluck with your research !")
            break

        # Show the top retrieved chunks before generating
        print("\n--- Retrieved context ---")
        print_top_results_and_scores(query=query, embeddings=embeddings)

        print("--- Generating answer ---")
        answer = ask(query)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
