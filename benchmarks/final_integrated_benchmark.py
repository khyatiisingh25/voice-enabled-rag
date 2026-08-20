import statistics
import time

from app.pipeline import pipeline


QUERIES = [
    "What is RAG?",
    "How does RAG work?",
    "What are embeddings?",
    "What does the system retrieve?",
    "What happens after retrieval?",
]


def percentile(values, percentile):
    values = sorted(values)

    index = int(
        (percentile / 100) * len(values)
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

    result = pipeline.query(query)

    total_ms = (
        time.perf_counter() - start
    ) * 1000

    return total_ms, result


def main():

    print("\n=== FINAL INTEGRATED RAG BENCHMARK ===")

    print("\nRunning warm-up...")

    warmup_time, warmup_result = run_query(
        "What is RAG?"
    )

    print(
        f"Warm-up latency: "
        f"{warmup_time:.2f} ms"
    )

    print(
        "Warm-up answer:",
        warmup_result["answer"],
    )

    print("\n=== STEADY-STATE ===")

    times = []
    fallback_count = 0

    for query in QUERIES:

        elapsed_ms, result = run_query(
            query
        )

        times.append(elapsed_ms)

        if result["fallback"]:
            fallback_count += 1

        print(
            f"\nQuestion: {query}"
        )

        print(
            f"Latency: {elapsed_ms:.2f} ms"
        )

        print(
            f"Fallback: "
            f"{result['fallback']}"
        )

        print(
            f"Validation: "
            f"{result['validation_reason']}"
        )

        print(
            f"Answer: "
            f"{result['answer']}"
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

    print(
        f"Fallback usage: "
        f"{fallback_count}/{len(QUERIES)}"
    )

    print(
        f"Fallback rate: "
        f"{(fallback_count / len(QUERIES)) * 100:.1f}%"
    )

    print(
        f"\nCold/warm-up request: "
        f"{warmup_time:.2f} ms"
    )


if __name__ == "__main__":
    main()
