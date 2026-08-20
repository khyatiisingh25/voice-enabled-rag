import statistics
import time

from app.pipeline import pipeline
from app.rag.embeddings import create_embeddings
from app.rag.generator import generate_answer


QUERIES = [
    "What is RAG?",
    "How does RAG work?",
    "What is retrieval augmented generation?",
    "What is the purpose of embeddings?",
    "How does the system create embeddings?",
    "What does FAISS do?",
    "How does vector search work?",
    "What is semantic search?",
    "How are documents retrieved?",
    "What is the role of the retriever?",
    "How does the system find relevant documents?",
    "What is a vector database?",
    "How does relevance filtering work?",
    "What happens when a query is unsupported?",
    "How does the system generate an answer?",
    "What makes an answer grounded?",
    "What is the RAG pipeline?",
    "How does the query move through the system?",
    "What are the sources returned by the API?",
    "How does the system handle irrelevant questions?",
]


def percentile(sorted_times, percentile):
    index = int((percentile / 100) * len(sorted_times)) - 1
    index = max(0, min(index, len(sorted_times) - 1))
    return sorted_times[index]


def benchmark():
    embedding_times = []
    retrieval_times = []
    generation_times = []
    total_times = []

    print("\n=== LATENCY BREAKDOWN ===")

    for query in QUERIES:

        # --------------------------------------------------
        # 1. Query embedding
        # --------------------------------------------------
        start = time.perf_counter()

        query_embedding = create_embeddings(
            pipeline.model,
            [query],
        )

        embedding_ms = (time.perf_counter() - start) * 1000

        # --------------------------------------------------
        # 2. FAISS retrieval
        # --------------------------------------------------
        start = time.perf_counter()

        retrieved_documents = pipeline.retriever.search(
            query_embedding,
            top_k=3,
        )

        retrieval_ms = (time.perf_counter() - start) * 1000

        # --------------------------------------------------
        # 3. LLM generation
        # --------------------------------------------------
        start = time.perf_counter()

        generate_answer(
            query,
            retrieved_documents,
        )

        generation_ms = (time.perf_counter() - start) * 1000

        # --------------------------------------------------
        # 4. Total query latency
        # --------------------------------------------------
        total_ms = (
            embedding_ms
            + retrieval_ms
            + generation_ms
        )

        embedding_times.append(embedding_ms)
        retrieval_times.append(retrieval_ms)
        generation_times.append(generation_ms)
        total_times.append(total_ms)

        print(
            f"\n{query}"
            f"\n  Embedding : {embedding_ms:.2f} ms"
            f"\n  Retrieval : {retrieval_ms:.2f} ms"
            f"\n  Generation: {generation_ms:.2f} ms"
            f"\n  Total     : {total_ms:.2f} ms"
        )

    print("\n=== LATENCY SUMMARY ===")

    stages = {
        "Embedding": embedding_times,
        "Retrieval": retrieval_times,
        "Generation": generation_times,
        "Total": total_times,
    }

    for name, times in stages.items():
        ordered = sorted(times)

        print(f"\n{name}")
        print(f"  P50 : {statistics.median(ordered):.2f} ms")
        print(f"  P70 : {percentile(ordered, 70):.2f} ms")
        print(f"  P100: {max(ordered):.2f} ms")


if __name__ == "__main__":
    benchmark()
