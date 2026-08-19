import pytest

from eval_engine.metrics.retrieval_rankers import (
    BM25Metric,
    ReciprocalRankFusionMetric,
    MMRMetric,
    LearningToRankHeuristicMetric,
)
from eval_engine.metrics import registry

@pytest.fixture
def sample_data():
    question = "What is retrieval augmented generation"

    contexts = [
        "Cats are animals that like sleeping.",
        "Retrieval augmented generation combines retrieval systems with large language models.",
        "The weather is sunny today.",
        "RAG improves factual grounding by retrieving relevant documents."
    ]

    answer = (
        "Retrieval augmented generation combines document retrieval "
        "with language model generation."
    )

    ground_truth = (
        "RAG retrieves relevant information before generating a response."
    )

    return question, contexts, answer, ground_truth


def test_bm25_returns_valid_score(sample_data):
    question, contexts, answer, gt = sample_data

    metric = BM25Metric()

    result = metric.score(
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=gt,
    )

    assert result.score >= 0
    assert "ranked_contexts" in result.raw
    assert len(result.raw["ranked_contexts"]) > 0


def test_bm25_ranks_relevant_context_first(sample_data):
    question, contexts, answer, gt = sample_data

    metric = BM25Metric()

    result = metric.score(
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=gt,
    )

    top_context = result.raw["ranked_contexts"][0]["context"]

    assert (
        "retrieval augmented generation"
        in top_context.lower()
    )


def test_rrf_fuses_rankings():
    metric = ReciprocalRankFusionMetric()

    rankings = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
    ]

    result = metric.score(
        question="",
        contexts=[],
        answer="",
        rankings=rankings,
    )

    fused = result.raw["fused_ranking"]

    assert len(fused) == 4

    # Doc 1 appears near top most often
    assert fused[0]["index"] == 1


def test_mmr_returns_diverse_documents(sample_data):
    question, contexts, answer, gt = sample_data

    metric = MMRMetric()

    result = metric.score(
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=gt,
        top_k=2,
        lambda_mult=0.7,
    )

    selected = result.raw["selected_contexts"]

    assert len(selected) == 2
    assert result.score >= 0


def test_ltr_scores_documents(sample_data):
    question, contexts, answer, gt = sample_data

    metric = LearningToRankHeuristicMetric()

    result = metric.score(
        question=question,
        contexts=contexts,
        answer=answer,
        ground_truth=gt,
    )

    ranked = result.raw["ranked_contexts"]

    assert len(ranked) == len(contexts)
    assert ranked[0]["score"] >= ranked[-1]["score"]


def test_empty_contexts_do_not_crash():
    metric = BM25Metric()

    result = metric.score(
        question="test",
        contexts=[],
        answer="test",
    )

    assert result.score == 0.0


def test_mmr_empty_contexts():
    metric = MMRMetric()

    result = metric.score(
        question="test",
        contexts=[],
        answer="test",
    )

    assert result.score == 0.0
    assert "selected_contexts" in result.raw


def test_ltr_without_ground_truth():
    metric = LearningToRankHeuristicMetric()

    result = metric.score(
        question="What is RAG",
        contexts=["RAG retrieves documents"],
        answer="RAG uses retrieval",
    )

    assert result.score >= 0

from eval_engine.metrics import registry


def test_bm25_registry_loads():
    metric = registry.get("bm25")
    assert metric.name == "bm25"


def test_rrf_registry_loads():
    metric = registry.get("rrf")
    assert metric.name == "rrf"


def test_mmr_registry_loads():
    metric = registry.get("mmr")
    assert metric.name == "mmr"


def test_ltr_registry_loads():
    metric = registry.get("ltr_heuristic")
    assert metric.name == "ltr_heuristic"