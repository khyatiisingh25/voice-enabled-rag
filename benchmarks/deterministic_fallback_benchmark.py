import statistics
import time

from app.pipeline import pipeline
from app.rag.embeddings import create_embeddings
from app.rag.validator import deterministic_fallback


QUERIES = [
    "What is RAG?",
    "How does RAG work?",
    "What are embeddings?",
    "What does the system retrieve?",
    "What happens after retrieval?",
]


def main():

    times = []

    print("=== PRODUCTION DETERMINISTIC FALLBACK ===")

    for query in QUERIES:

        embedding = create_embeddings(
            pipeline.model,
            [query],
        )

        results = pipeline.retriever.search(
            embedding,
            top_k=3,
        )

        start = time.perf_counter()

        answer = deterministic_fallback(
            query,
            results,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        times.append(elapsed_ms)

        print(f"\nQuestion: {query}")
        print(f"Answer:   {answer}")
        print(f"Latency:  {elapsed_ms:.4f} ms")

    print("\n=== RESULTS ===")

    print(
        f"P50: "
        f"{statistics.median(times):.4f} ms"
    )

    print(
        f"P100: "
        f"{max(times):.4f} ms"
    )


if __name__ == "__main__":
    main()
