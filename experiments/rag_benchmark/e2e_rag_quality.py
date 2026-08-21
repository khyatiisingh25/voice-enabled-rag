import json
import statistics
import time
from pathlib import Path

from app.rag.loader import load_documents
from app.rag.chunker import chunk_text
from app.rag.embeddings import (
    create_embedding_model,
    create_embeddings,
)
from app.rag.retriever import Retriever
from app.rag.validator import (
    deterministic_fallback,
    validate_answer,
    REFUSAL,
)


EVAL_FILE = Path(
    "experiments/rag_benchmark/msmarco_eval.json"
)

OUTPUT_FILE = Path(
    "experiments/rag_benchmark/e2e_rag_quality_results.json"
)

TOP_K_VALUES = [3, 10]
THRESHOLD = 0.20


def normalize(text):
    return set(
        word.lower()
        for word in text.split()
        if word.strip()
    )


def answer_correctness(answer, reference):
    """
    Lightweight benchmark correctness signal.

    Measures lexical overlap between the generated answer
    and the benchmark reference answer.
    """
    if answer == REFUSAL:
        return 0.0

    answer_words = normalize(answer)
    reference_words = normalize(reference)

    if not reference_words:
        return 0.0

    overlap = answer_words & reference_words

    return len(overlap) / len(reference_words)


def source_quality(results, relevant_indices):
    """
    Checks whether at least one retrieved source corresponds
    to a ground-truth relevant passage.
    """

    for rank, result in enumerate(results, start=1):
        source = result["source"]

        if source.startswith("passage_"):
            try:
                passage_index = int(
                    source.split("_", 1)[1]
                )
            except ValueError:
                continue

            if passage_index in relevant_indices:
                return {
                    "relevant_source_found": True,
                    "first_relevant_rank": rank,
                }

    return {
        "relevant_source_found": False,
        "first_relevant_rank": None,
    }


def evaluate_answer(
    question,
    answer,
    retrieved_documents,
    reference_answer,
):
    context = "\n".join(
        document["text"]
        for document in retrieved_documents
    )

    valid, validation_reason = validate_answer(
        question,
        answer,
        context,
    )

    correctness = answer_correctness(
        answer,
        reference_answer,
    )

    grounded = valid

    hallucinated = (
        answer != REFUSAL
        and not grounded
    )

    return {
        "correctness_score": correctness,
        "grounded": grounded,
        "hallucinated": hallucinated,
        "validation_reason": validation_reason,
    }


def build_retriever(model, samples):
    documents = []

    for sample in samples:

        for passage_index, passage in enumerate(
            sample["passages"]
        ):

            chunks = chunk_text(passage)

            for chunk_index, chunk in enumerate(
                chunks
            ):
                documents.append(
                    {
                        "text": chunk,
                        "source": (
                            f"passage_{passage_index}"
                        ),
                        "chunk_index": chunk_index,
                    }
                )

    embeddings = create_embeddings(
        model,
        [
            document["text"]
            for document in documents
        ],
    )

    return Retriever(
        embeddings,
        documents,
    )


def evaluate_top_k(
    model,
    retriever,
    samples,
    top_k,
):
    results = []
    latencies = []

    for sample_index, sample in enumerate(samples):

        query = sample["query"]

        relevant_indices = {
            index
            for index, label in enumerate(
                sample["is_selected"]
            )
            if label == 1
        }

        query_embedding = create_embeddings(
            model,
            [query],
        )

        start = time.perf_counter()

        retrieved = retriever.search(
            query_embedding,
            top_k=top_k,
            score_threshold=THRESHOLD,
        )

        answer = deterministic_fallback(
            query,
            retrieved,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        quality = evaluate_answer(
            query,
            answer,
            retrieved,
            sample.get("answer", ""),
        )

        sources = source_quality(
            retrieved,
            relevant_indices,
        )

        results.append(
            {
                "sample_index": sample_index,
                "query_id": sample.get("query_id"),
                "query": query,
                "reference_answer": sample.get(
                    "answer",
                    "",
                ),
                "answer": answer,
                "retrieved_sources": [
                    {
                        "rank": rank,
                        "source": item["source"],
                        "score": item["score"],
                    }
                    for rank, item in enumerate(
                        retrieved,
                        start=1,
                    )
                ],
                "source_quality": sources,
                **quality,
            }
        )

        latencies.append(elapsed_ms)

    non_refusal = [
        item
        for item in results
        if item["answer"] != REFUSAL
    ]

    summary = {
        "queries": len(results),
        "mean_correctness": statistics.mean(
            item["correctness_score"]
            for item in results
        ),
        "grounded_rate": (
            sum(
                item["grounded"]
                for item in results
            )
            / len(results)
        ),
        "hallucination_rate": (
            sum(
                item["hallucinated"]
                for item in results
            )
            / len(results)
        ),
        "relevant_source_rate": (
            sum(
                item["source_quality"][
                    "relevant_source_found"
                ]
                for item in results
            )
            / len(results)
        ),
        "non_refusal_rate": (
            len(non_refusal)
            / len(results)
        ),
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": statistics.quantiles(
                latencies,
                n=20,
            )[18],
            "min": min(latencies),
            "max": max(latencies),
        },
    }

    return {
        "summary": summary,
        "results": results,
    }


def compare_results(old, new):
    regressions = []

    old_results = {
        item["query_id"]: item
        for item in old["results"]
    }

    new_results = {
        item["query_id"]: item
        for item in new["results"]
    }

    for query_id, old_item in old_results.items():

        new_item = new_results.get(query_id)

        if not new_item:
            continue

        worse = []

        if (
            new_item["correctness_score"]
            < old_item["correctness_score"]
        ):
            worse.append("correctness")

        if (
            old_item["grounded"]
            and not new_item["grounded"]
        ):
            worse.append("groundedness")

        if (
            old_item["source_quality"][
                "relevant_source_found"
            ]
            and not new_item["source_quality"][
                "relevant_source_found"
            ]
        ):
            worse.append("source_quality")

        if (
            not old_item["hallucinated"]
            and new_item["hallucinated"]
        ):
            worse.append("hallucination")

        if worse:
            regressions.append(
                {
                    "query_id": query_id,
                    "query": old_item["query"],
                    "reasons": worse,
                    "top_k_3": {
                        "answer": old_item["answer"],
                        "correctness": old_item[
                            "correctness_score"
                        ],
                        "grounded": old_item[
                            "grounded"
                        ],
                    },
                    "top_k_10": {
                        "answer": new_item["answer"],
                        "correctness": new_item[
                            "correctness_score"
                        ],
                        "grounded": new_item[
                            "grounded"
                        ],
                    },
                }
            )

    return regressions


def main():

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        samples = json.load(file)

    model = create_embedding_model()

    retriever = build_retriever(
        model,
        samples,
    )

    all_results = {}

    for top_k in TOP_K_VALUES:

        print(
            f"\nEvaluating top_k={top_k}"
        )

        evaluation = evaluate_top_k(
            model,
            retriever,
            samples,
            top_k,
        )

        all_results[str(top_k)] = evaluation

        print(
            json.dumps(
                evaluation["summary"],
                indent=2,
            )
        )

    regressions = compare_results(
        all_results["3"],
        all_results["10"],
    )

    output = {
        "configuration": {
            "dataset": str(EVAL_FILE),
            "queries": len(samples),
            "threshold": THRESHOLD,
            "top_k_comparison": [3, 10],
            "production_code_modified": False,
        },
        "results": all_results,
        "top_k_10_regressions": regressions,
    }

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
        f"\nResults saved to: {OUTPUT_FILE}"
    )

    print(
        f"top_k=10 regressions: "
        f"{len(regressions)}"
    )


if __name__ == "__main__":
    main()