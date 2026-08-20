import json

from experiments.rag_benchmark.evaluator import reciprocal_rank


def test_reciprocal_rank_first_relevant():
    ranked_indices = [0, 2, 1, 3]
    relevant_indices = {0}

    assert reciprocal_rank(
        ranked_indices,
        relevant_indices,
    ) == 1.0


def test_reciprocal_rank_second_relevant():
    ranked_indices = [2, 0, 1, 3]
    relevant_indices = {0}

    assert reciprocal_rank(
        ranked_indices,
        relevant_indices,
    ) == 0.5


def test_reciprocal_rank_no_relevant_result():
    ranked_indices = [2, 3, 1]
    relevant_indices = {0}

    assert reciprocal_rank(
        ranked_indices,
        relevant_indices,
    ) == 0.0


def test_reciprocal_rank_multiple_relevant_results():
    ranked_indices = [3, 2, 0, 1]
    relevant_indices = {0, 1}

    assert reciprocal_rank(
        ranked_indices,
        relevant_indices,
    ) == 1 / 3


def test_retrieval_results_file_structure():
    with open(
        "experiments/rag_benchmark/retrieval_results.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert "metrics" in data
    assert "results" in data

    metrics = data["metrics"]

    assert metrics["num_queries"] == 100
    assert 0.0 <= metrics["R@1"] <= 1.0
    assert 0.0 <= metrics["R@3"] <= 1.0
    assert 0.0 <= metrics["R@5"] <= 1.0
    assert 0.0 <= metrics["MRR"] <= 1.0

    assert len(data["results"]) == 100


def test_retrieval_result_structure():
    with open(
        "experiments/rag_benchmark/retrieval_results.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    result = data["results"][0]

    assert "query_id" in result
    assert "query" in result
    assert "hit_at_1" in result
    assert "hit_at_3" in result
    assert "hit_at_5" in result
    assert "mrr" in result
    assert "grounded_at_1" in result
    assert "relevant_passage_indices" in result
    assert "ranked_passages" in result

    assert isinstance(result["ranked_passages"], list)