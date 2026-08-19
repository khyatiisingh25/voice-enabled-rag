import argparse
import json
from pathlib import Path

from app.rag.chunking_strategies import chunk_with_strategy
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever


STRATEGIES = [
    "500/50",
    "300/50",
    "800/100",
    "sentence",
    "metadata",
]


def recall_at_k(relevant_ranks, k):
    return 1.0 if any(rank <= k for rank in relevant_ranks) else 0.0


def reciprocal_rank(relevant_ranks):
    if not relevant_ranks:
        return 0.0

    return 1.0 / min(relevant_ranks)


def get_passage_index(result):
    """
    Recover the original MSMARCO passage index from the
    source field returned by the existing Retriever.
    """
    source = result.get("source", "")

    if source.startswith("passage_"):
        try:
            return int(source.split("_", 1)[1])
        except ValueError:
            return None

    return None


def evaluate_query(results, relevant_passage_indices):
    relevant_ranks = []

    for rank, result in enumerate(results, start=1):
        passage_index = get_passage_index(result)

        if passage_index in relevant_passage_indices:
            relevant_ranks.append(rank)

    return {
        "recall@1": recall_at_k(relevant_ranks, 1),
        "recall@3": recall_at_k(relevant_ranks, 3),
        "recall@5": recall_at_k(relevant_ranks, 5),
        "mrr": reciprocal_rank(relevant_ranks),
    }


def evaluate_strategy(samples, strategy, model):
    all_metrics = []
    ranked_results = []

    for sample_index, sample in enumerate(samples):
        query = sample["query"]
        passages = sample["passages"]
        labels = sample["is_selected"]

        relevant_passage_indices = {
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
            [query],
        )

        results = retriever.search(
            query_embedding,
            top_k=5,
        )

        metrics = evaluate_query(
            results,
            relevant_passage_indices,
        )

        all_metrics.append(metrics)

        ranked_results.append(
            {
                "sample_index": sample_index,
                "query_id": sample.get("query_id"),
                "query": query,
                "results": [
                    {
                        "rank": rank,
                        "source": result["source"],
                        "score": result["score"],
                        "text": result["text"],
                    }
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    count = len(all_metrics)

    if count == 0:
        summary = {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
        }
    else:
        summary = {
            metric: sum(item[metric] for item in all_metrics) / count
            for metric in [
                "recall@1",
                "recall@3",
                "recall@5",
                "mrr",
            ]
        }

    return {
        "metrics": summary,
        "queries_evaluated": count,
        "ranked_results": ranked_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval metrics across chunking strategies."
    )

    parser.add_argument(
        "--eval-data",
        required=True,
        help="Path to the MSMARCO-XI evaluation JSON file.",
    )

    parser.add_argument(
        "--output",
        default="benchmarks/retrieval_results.json",
        help="Path for the evaluation results JSON.",
    )

    args = parser.parse_args()

    eval_path = Path(args.eval_data)

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {eval_path}"
        )

    with eval_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    model = create_embedding_model()

    results = {}

    for strategy in STRATEGIES:
        print(f"\nEvaluating: {strategy}")

        results[strategy] = evaluate_strategy(
            samples,
            strategy,
            model,
        )

        print(
            results[strategy]["metrics"]
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
