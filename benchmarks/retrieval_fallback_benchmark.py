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


def percentile(values, p):
    values = sorted(values)

    index = int(
        (p / 100) * len(values)
    ) - 1

    index = max(
        0,
        min(
            len(values) - 1,
            index,
        ),
    )

    return values[index]


def run_query(query):

    start = time.perf_counter()

    embedding = create_embeddings(
        pipeline.model,
        [query],
    )

    embedding_ms = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    results = pipeline.retriever.search(
        embedding,
        top_k=3,
    )

    retrieval_ms = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    answer = deterministic_fallback(
        query,
        results,
    )

    fallback_ms = (
        time.perf_counter() - start
    ) * 1000

    total_ms = (
        embedding_ms
        + retrieval_ms
        + fallback_ms
    )

    return (
        embedding_ms,
        retrieval_ms,
        fallback_ms,
        total_ms,
        answer,
    )


def main():

    print(
        "\n=== RETRIEVAL + DETERMINISTIC FALLBACK ==="
    )

    # Warm-up embedding model
    print("\nRunning warm-up...")

    run_query("What is RAG?")

    times = []

    for query in QUERIES:

        (
            embedding_ms,
            retrieval_ms,
            fallback_ms,
            total_ms,
            answer,
        ) = run_query(query)

        times.append(total_ms)

        print(
            f"\nQuestion: {query}"
        )

        print(
            f"Embedding : "
            f"{embedding_ms:.2f} ms"
        )

        print(
            f"FAISS     : "
            f"{retrieval_ms:.2f} ms"
        )

        print(
            f"Fallback  : "
            f"{fallback_ms:.4f} ms"
        )

        print(
            f"TOTAL     : "
            f"{total_ms:.2f} ms"
        )

        print(
            f"Answer: {answer}"
        )

    print("\n=== RESULTS ===")

    print(
        f"P50: "
        f"{statistics.median(times):.2f} ms"
    )

    print(
        f"P70: "
        f"{percentile(times, 70):.2f} ms"
    )

    print(
        f"P100: "
        f"{max(times):.2f} ms"
    )


if __name__ == "__main__":
    main()
