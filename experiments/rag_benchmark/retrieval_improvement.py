import json
from pathlib import Path

from app.rag.chunking_strategies import chunk_with_strategy
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever


EVAL_FILE = Path("experiments/rag_benchmark/msmarco_eval.json")
OUTPUT_FILE = Path(
    "experiments/rag_benchmark/retrieval_improvement_results.json"
)

STRATEGIES = [
    "500/50",
    "300/50",
    "800/100",
    "sentence",
    "metadata",
]

TOP_K_VALUES = [3, 5, 10]
THRESHOLD_VALUES = [0.10, 0.20, 0.30]


def evaluate_query(results, relevant_indices):
    relevant_ranks = []

    for rank, result in enumerate(results, start=1):
        source = result.get("source", "")

        if source.startswith("passage_"):
            try:
                passage_index = int(source.split("_", 1)[1])
            except ValueError:
                continue

            if passage_index in relevant_indices:
                relevant_ranks.append(rank)

    if relevant_ranks:
        first_rank = min(relevant_ranks)
        mrr = 1.0 / first_rank
    else:
        first_rank = None
        mrr = 0.0

    return {
        "recall@1": float(first_rank is not None and first_rank <= 1),
        "recall@3": float(first_rank is not None and first_rank <= 3),
        "recall@5": float(first_rank is not None and first_rank <= 5),
        "mrr": mrr,
    }


def average_metrics(metrics):
    if not metrics:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
        }

    return {
        metric: sum(item[metric] for item in metrics) / len(metrics)
        for metric in [
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr",
        ]
    }


def evaluate_configuration(
    samples,
    strategy,
    top_k,
    threshold,
    model,
):
    metrics = []

    for sample in samples:
        passages = sample["passages"]
        labels = sample["is_selected"]

        relevant_indices = {
            index
            for index, label in enumerate(labels)
            if label == 1
        }

        documents = []

        for passage_index, passage in enumerate(passages):
            chunks = chunk_with_strategy(
                passage,
                strategy,
                source=f"passage_{passage_index}",
            )

            for chunk_index, chunk in enumerate(chunks):
                documents.append(
                    {
                        "text": chunk["text"],
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

        results = retriever.search(
            query_embedding,
            top_k=top_k,
            score_threshold=threshold,
        )

        metrics.append(
            evaluate_query(
                results,
                relevant_indices,
            )
        )

    return {
        "queries_evaluated": len(metrics),
        "metrics": average_metrics(metrics),
    }


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError("Evaluation dataset must contain a JSON list.")

    model = create_embedding_model()

    results = {}

    total = (
        len(STRATEGIES)
        * len(TOP_K_VALUES)
        * len(THRESHOLD_VALUES)
    )

    current = 0

    for strategy in STRATEGIES:
        results[strategy] = {}

        for top_k in TOP_K_VALUES:
            results[strategy][str(top_k)] = {}

            for threshold in THRESHOLD_VALUES:
                current += 1

                print(
                    f"[{current}/{total}] "
                    f"{strategy} | top_k={top_k} | "
                    f"threshold={threshold}"
                )

                result = evaluate_configuration(
                    samples,
                    strategy,
                    top_k,
                    threshold,
                    model,
                )

                results[strategy][str(top_k)][str(threshold)] = result

                print(result["metrics"])

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()