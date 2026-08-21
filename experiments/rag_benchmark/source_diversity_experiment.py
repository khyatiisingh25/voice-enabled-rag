import json
from pathlib import Path

from app.rag.chunking_strategies import chunk_with_strategy
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever


EVAL_FILE = Path("experiments/rag_benchmark/msmarco_eval.json")
OUTPUT_FILE = Path(
    "experiments/rag_benchmark/source_diversity_results.json"
)

STRATEGY = "800/100"
THRESHOLD = 0.20

BASELINE_TOP_K = 3
CANDIDATE_POOL = 10
FINAL_TOP_K = 3


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


def source_diverse_select(results, final_top_k):
    selected = []
    seen_sources = set()

    # First pass: prefer different source passages.
    for result in results:
        source = result["source"]

        if source in seen_sources:
            continue

        selected.append(result)
        seen_sources.add(source)

        if len(selected) == final_top_k:
            return selected

    # Fallback: if there are not enough unique sources,
    # fill remaining positions using the original ranking.
    selected_ids = {id(result) for result in selected}

    for result in results:
        if id(result) in selected_ids:
            continue

        selected.append(result)

        if len(selected) == final_top_k:
            break

    return selected


def average_metrics(items):
    if not items:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
        }

    return {
        metric: sum(item[metric] for item in items) / len(items)
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

    baseline_metrics = []
    candidate_metrics = []

    for sample in samples:
        relevant_indices = {
            index
            for index, label in enumerate(sample["is_selected"])
            if label == 1
        }

        documents = []

        for passage_index, passage in enumerate(sample["passages"]):
            chunks = chunk_with_strategy(
                passage,
                STRATEGY,
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

        embeddings = create_embeddings(
            model,
            [document["text"] for document in documents],
        )

        retriever = Retriever(
            embeddings,
            documents,
        )

        query_embedding = create_embeddings(
            model,
            [sample["query"]],
        )

        # Current production-style baseline.
        baseline_results = retriever.search(
            query_embedding,
            top_k=BASELINE_TOP_K,
            score_threshold=THRESHOLD,
        )

        # Candidate: retrieve a larger pool, then enforce
        # source diversity before returning the final top-k.
        candidate_results = retriever.search(
            query_embedding,
            top_k=CANDIDATE_POOL,
            score_threshold=THRESHOLD,
        )

        diverse_results = source_diverse_select(
            candidate_results,
            FINAL_TOP_K,
        )

        baseline_metrics.append(
            evaluate(
                baseline_results,
                relevant_indices,
            )
        )

        candidate_metrics.append(
            evaluate(
                diverse_results,
                relevant_indices,
            )
        )

    output = {
        "configuration": {
            "strategy": STRATEGY,
            "threshold": THRESHOLD,
            "baseline_top_k": BASELINE_TOP_K,
            "candidate_pool": CANDIDATE_POOL,
            "candidate_final_top_k": FINAL_TOP_K,
        },
        "queries_evaluated": len(baseline_metrics),
        "baseline": {
            "metrics": average_metrics(baseline_metrics),
        },
        "source_diverse_candidate": {
            "metrics": average_metrics(candidate_metrics),
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Source Diversity Experiment ===")
    print("Baseline:")
    print(output["baseline"]["metrics"])

    print("\nSource-diverse candidate:")
    print(
        output["source_diverse_candidate"]["metrics"]
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()