import json
import time
from pathlib import Path

from app.rag.chunker import chunk_text
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever


EVAL_FILE = Path("experiments/rag_benchmark/msmarco_eval.json")
OUTPUT_FILE = Path(
    "experiments/rag_benchmark/production_topk_results.json"
)

THRESHOLD = 0.20
TOP_K_VALUES = [3, 5, 10]


def passage_index_from_source(source):
    if not source.startswith("passage_"):
        return None

    try:
        return int(source.split("_", 1)[1])
    except ValueError:
        return None


def evaluate(results, relevant_indices):
    relevant_ranks = []

    for rank, result in enumerate(results, start=1):
        passage_index = passage_index_from_source(
            result["source"]
        )

        if passage_index in relevant_indices:
            relevant_ranks.append(rank)

    if not relevant_ranks:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
        }

    first_rank = min(relevant_ranks)

    return {
        "recall@1": float(first_rank <= 1),
        "recall@3": float(first_rank <= 3),
        "recall@5": float(first_rank <= 5),
        "mrr": 1.0 / first_rank,
    }


def average_metrics(metrics):
    return {
        metric: sum(item[metric] for item in metrics) / len(metrics)
        for metric in [
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr",
        ]
    }


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    model = create_embedding_model()

    metric_results = {
        top_k: []
        for top_k in TOP_K_VALUES
    }

    latency_results = {
        top_k: []
        for top_k in TOP_K_VALUES
    }

    queries_evaluated = 0

    for sample in samples:
        documents = []

        for passage_index, passage in enumerate(
            sample["passages"]
        ):
            # Use the same chunking function as production.
            chunks = chunk_text(passage)

            for chunk_index, chunk in enumerate(chunks):
                documents.append(
                    {
                        "text": chunk,
                        "source": f"passage_{passage_index}",
                        "chunk_index": chunk_index,
                    }
                )

        if not documents:
            continue

        document_embeddings = create_embeddings(
            model,
            [document["text"] for document in documents],
        )

        retriever = Retriever(
            document_embeddings,
            documents,
        )

        query_embedding = create_embeddings(
            model,
            [sample["query"]],
        )

        relevant_indices = {
            index
            for index, label in enumerate(
                sample["is_selected"]
            )
            if label == 1
        }

        for top_k in TOP_K_VALUES:
            start = time.perf_counter()

            results = retriever.search(
                query_embedding,
                top_k=top_k,
                score_threshold=THRESHOLD,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            metric_results[top_k].append(
                evaluate(
                    results,
                    relevant_indices,
                )
            )

            latency_results[top_k].append(
                elapsed_ms
            )

        queries_evaluated += 1

    output = {
        "configuration": {
            "chunking": "production chunk_text()",
            "threshold": THRESHOLD,
            "top_k_values": TOP_K_VALUES,
        },
        "queries_evaluated": queries_evaluated,
        "results": {},
    }

    for top_k in TOP_K_VALUES:
        latencies = latency_results[top_k]

        output["results"][str(top_k)] = {
            "metrics": average_metrics(
                metric_results[top_k]
            ),
            "retrieval_latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
            },
        }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\n=== Production Top-k Experiment ===")
    print(
        f"Queries evaluated: {queries_evaluated}"
    )

    for top_k in TOP_K_VALUES:
        data = output["results"][str(top_k)]

        print(
            f"\ntop_k={top_k}"
        )
        print(
            f"Metrics: {data['metrics']}"
        )
        print(
            f"Latency: {data['retrieval_latency_ms']}"
        )

    print(
        f"\nResults saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()