import json
from pathlib import Path

import numpy as np

from app.rag.embeddings import create_embedding_model, create_embeddings


EVAL_FILE = Path("experiments/rag_benchmark/msmarco_eval.json")
RESULTS_FILE = Path("experiments/rag_benchmark/retrieval_results.json")


def reciprocal_rank(ranked_indices: list[int], relevant_indices: set[int]) -> float:
    for rank, index in enumerate(ranked_indices, start=1):
        if index in relevant_indices:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    eval_file: Path = EVAL_FILE,
    results_file: Path = RESULTS_FILE,
) -> dict:
    with eval_file.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty")

    model = create_embedding_model()

    all_results = []

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    reciprocal_ranks = []

    for item in dataset:
        query = item["query"]
        passages = item["passages"]
        is_selected = item["is_selected"]

        if len(passages) != len(is_selected):
            raise ValueError(
                f"Passage/label mismatch for query_id={item['query_id']}"
            )

        relevant_indices = {
            index
            for index, selected in enumerate(is_selected)
            if selected == 1
        }

        if not relevant_indices:
            raise ValueError(
                f"No relevant passage for query_id={item['query_id']}"
            )

        query_embedding = create_embeddings(
            model,
            [query],
        )[0]

        passage_embeddings = create_embeddings(
            model,
            passages,
        )

        scores = np.asarray(passage_embeddings) @ np.asarray(query_embedding)

        ranked_indices = np.argsort(scores)[::-1].tolist()

        top_1 = ranked_indices[:1]
        top_3 = ranked_indices[:3]
        top_5 = ranked_indices[:5]

        hit_at_1 = int(
            any(index in relevant_indices for index in top_1)
        )
        hit_at_3 = int(
            any(index in relevant_indices for index in top_3)
        )
        hit_at_5 = int(
            any(index in relevant_indices for index in top_5)
        )

        mrr = reciprocal_rank(
            ranked_indices,
            relevant_indices,
        )

        recall_at_1.append(hit_at_1)
        recall_at_3.append(hit_at_3)
        recall_at_5.append(hit_at_5)
        reciprocal_ranks.append(mrr)

        ranked_passages = []

        for rank, passage_index in enumerate(ranked_indices, start=1):
            ranked_passages.append(
                {
                    "rank": rank,
                    "passage_index": passage_index,
                    "score": float(scores[passage_index]),
                    "is_selected": is_selected[passage_index],
                    "text": passages[passage_index],
                }
            )

        all_results.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "mrr": mrr,
                "grounded_at_1": bool(hit_at_1),
                "relevant_passage_indices": sorted(relevant_indices),
                "ranked_passages": ranked_passages,
            }
        )

    metrics = {
        "num_queries": len(dataset),
        "R@1": float(np.mean(recall_at_1)),
        "R@3": float(np.mean(recall_at_3)),
        "R@5": float(np.mean(recall_at_5)),
        "MRR": float(np.mean(reciprocal_ranks)),
        "grounding_rate_at_1": float(np.mean(recall_at_1)),
    }

    output = {
        "metrics": metrics,
        "results": all_results,
    }

    with results_file.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output


if __name__ == "__main__":
    output = evaluate_retrieval()

    print("\n=== Retrieval Benchmark ===")
    print(f"Queries : {output['metrics']['num_queries']}")
    print(f"R@1    : {output['metrics']['R@1']:.4f}")
    print(f"R@3    : {output['metrics']['R@3']:.4f}")
    print(f"R@5    : {output['metrics']['R@5']:.4f}")
    print(f"MRR    : {output['metrics']['MRR']:.4f}")
    print(
        "Grounding@1 : "
        f"{output['metrics']['grounding_rate_at_1']:.4f}"
    )