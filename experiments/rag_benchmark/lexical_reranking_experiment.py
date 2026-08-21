import json
import re
from pathlib import Path

from app.rag.chunking_strategies import chunk_with_strategy
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever


EVAL_FILE = Path("experiments/rag_benchmark/msmarco_eval.json")
OUTPUT_FILE = Path(
    "experiments/rag_benchmark/lexical_reranking_results.json"
)

STRATEGY = "800/100"
THRESHOLD = 0.20
CANDIDATE_TOP_K = 10
FINAL_TOP_K = 3

# Controlled lexical-reranking weights.
# semantic_weight + lexical_weight = 1.
WEIGHT_CONFIGS = [
    {"semantic": 1.00, "lexical": 0.00},
    {"semantic": 0.90, "lexical": 0.10},
    {"semantic": 0.80, "lexical": 0.20},
    {"semantic": 0.70, "lexical": 0.30},
]


def tokenize(text):
    return set(
        re.findall(
            r"\b[a-z0-9]+\b",
            text.lower(),
        )
    )


def lexical_overlap(query, document):
    query_tokens = tokenize(query)
    document_tokens = tokenize(document)

    if not query_tokens:
        return 0.0

    return len(query_tokens & document_tokens) / len(query_tokens)


def rerank(results, query, semantic_weight, lexical_weight):
    reranked = []

    for result in results:
        lexical_score = lexical_overlap(
            query,
            result["text"],
        )

        semantic_score = float(result["score"])

        combined_score = (
            semantic_weight * semantic_score
            + lexical_weight * lexical_score
        )

        updated = dict(result)
        updated["semantic_score"] = semantic_score
        updated["lexical_score"] = lexical_score
        updated["combined_score"] = combined_score

        reranked.append(updated)

    reranked.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return reranked[:FINAL_TOP_K]


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


def main():
    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        samples = json.load(file)

    model = create_embedding_model()

    results = {
        f"{config['semantic']:.2f}/{config['lexical']:.2f}": []
        for config in WEIGHT_CONFIGS
    }

    queries_evaluated = 0

    for sample in samples:
        relevant_indices = {
            index
            for index, label in enumerate(sample["is_selected"])
            if label == 1
        }

        documents = []

        for passage_index, passage in enumerate(
            sample["passages"]
        ):
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

        candidates = retriever.search(
            query_embedding,
            top_k=CANDIDATE_TOP_K,
            score_threshold=THRESHOLD,
        )

        queries_evaluated += 1

        for config in WEIGHT_CONFIGS:
            key = (
                f"{config['semantic']:.2f}/"
                f"{config['lexical']:.2f}"
            )

            reranked = rerank(
                candidates,
                sample["query"],
                config["semantic"],
                config["lexical"],
            )

            results[key].append(
                evaluate(
                    reranked,
                    relevant_indices,
                )
            )

    output = {
        "configuration": {
            "strategy": STRATEGY,
            "candidate_top_k": CANDIDATE_TOP_K,
            "final_top_k": FINAL_TOP_K,
            "threshold": THRESHOLD,
        },
        "queries_evaluated": queries_evaluated,
        "weight_results": {},
    }

    for key, metrics in results.items():
        output["weight_results"][key] = {
            "semantic_weight": float(key.split("/")[0]),
            "lexical_weight": float(key.split("/")[1]),
            "metrics": average_metrics(metrics),
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

    print("\n=== Lexical Reranking Experiment ===")
    print(f"Queries evaluated: {queries_evaluated}")

    for key, data in output["weight_results"].items():
        print(
            f"{key}: "
            f"{data['metrics']}"
        )

    print(
        f"\nResults saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()