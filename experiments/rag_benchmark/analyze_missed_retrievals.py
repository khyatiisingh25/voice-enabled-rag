import json
from pathlib import Path

from app.rag.chunking_strategies import chunk_with_strategy
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever


EVAL_FILE = Path("experiments/rag_benchmark/msmarco_eval.json")
OUTPUT_FILE = Path(
    "experiments/rag_benchmark/missed_retrievals.json"
)

STRATEGY = "800/100"
TOP_K = 5
THRESHOLD = 0.20


def get_passage_index(source):
    if not source.startswith("passage_"):
        return None

    try:
        return int(source.split("_", 1)[1])
    except ValueError:
        return None


def main():
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    model = create_embedding_model()

    missed_queries = []

    for sample_index, sample in enumerate(samples):
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
            top_k=TOP_K,
            score_threshold=THRESHOLD,
        )

        ranked_passages = []

        for rank, result in enumerate(results, start=1):
            passage_index = get_passage_index(result["source"])

            ranked_passages.append(
                {
                    "rank": rank,
                    "passage_index": passage_index,
                    "score": result["score"],
                }
            )

        retrieved_indices = {
            item["passage_index"]
            for item in ranked_passages
        }

        if not relevant_indices.intersection(retrieved_indices):
            missed_queries.append(
                {
                    "sample_index": sample_index,
                    "query_id": sample.get("query_id"),
                    "query": sample["query"],
                    "relevant_passage_indices": sorted(
                        relevant_indices
                    ),
                    "ranked_results": ranked_passages,
                }
            )

    output = {
        "configuration": {
            "strategy": STRATEGY,
            "top_k": TOP_K,
            "threshold": THRESHOLD,
        },
        "queries_evaluated": len(samples),
        "missed_queries": len(missed_queries),
        "miss_rate": (
            len(missed_queries) / len(samples)
            if samples
            else 0.0
        ),
        "missed": missed_queries,
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

    print(
        f"Queries evaluated: {len(samples)}"
    )
    print(
        f"Missed queries: {len(missed_queries)}"
    )
    print(
        f"Miss rate: {output['miss_rate']:.2%}"
    )
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()