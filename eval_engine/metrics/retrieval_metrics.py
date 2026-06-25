"""
eval_engine/metrics/retrieval_metrics.py

Retrieval Effectiveness Metrics — Recall@K, MRR, nDCG@K

SOURCE: Consolidation RAG eval framework (your code), UNCHANGED.
ADDITION: BaseMetric adapter classes wrap your functions so they can be
          invoked via CLI and registered in MetricsRegistry.
          Your original functions remain importable directly.

DCG FORMULA NOTE (for Consolidation Eval Suite methodology section):
    This module uses the exponential DCG formulation:
        DCG = sum((2^rel - 1) / log2(i+1))
    This is the standard TREC/IR definition and differs from RAGAS's
    internal linear formulation (rel / log2(i+2)).
    All NDCG@K values in Consolidation Eval Suite results use this exponential formula.
    RAGAS grounding scores use RAGAS's internal formula.
    These are reported in separate columns and must not be averaged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from eval_engine.schemas import RetrievalCase, RetrievalResult
from eval_engine.metrics.base import BaseMetric, MetricResult


# =============================================================================
# YOUR ORIGINAL CODE — VERBATIM (do not modify)
# =============================================================================

def _case_map(cases: list[RetrievalCase]) -> dict[str, RetrievalCase]:
    """Map benchmark cases by query_id for lookup."""
    # Convert the list of benchmark cases into a dictionary.
    # This lets us quickly look up the correct answer set for a query_id.
    return {case.query_id: case for case in cases}


def recall_at_k(
    cases: list[RetrievalCase],
    results: list[RetrievalResult],
    k: int,
) -> float:
    """
    Compute query-level Recall@K.
    A query is counted as successful if at least one relevant item appears within the first K retrieved results.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if not cases:
        return 0.0

    cases_by_id = _case_map(cases)
    hits = 0

    for result in results:
        case = cases_by_id.get(result.query_id)
        if case is None:
            continue
        top_k = set(result.retrieved_ids[:k])
        if top_k.intersection(case.relevant_ids):
            hits += 1

    return hits / len(cases)


def mean_reciprocal_rank(
    cases: list[RetrievalCase],
    results: list[RetrievalResult],
) -> float:
    """
    Compute Mean Reciprocal Rank across benchmark queries.
    MRR rewards systems that place the first relevant result higher in the ranked retrieval list.
    """
    if not cases:
        return 0.0

    cases_by_id = _case_map(cases)
    reciprocal_ranks: list[float] = []

    for result in results:
        case = cases_by_id.get(result.query_id)
        if case is None:
            continue

        rank_score = 0.0
        for index, retrieved_id in enumerate(result.retrieved_ids, start=1):
            if retrieved_id in case.relevant_ids:
                rank_score = 1.0 / index
                break

        reciprocal_ranks.append(rank_score)

    missing = len(cases) - len(reciprocal_ranks)
    reciprocal_ranks.extend([0.0] * max(missing, 0))

    return sum(reciprocal_ranks) / len(cases)


def ndcg_at_k(
    relevance_by_query: Mapping[str, Mapping[str, float]],
    results: list[RetrievalResult],
    k: int,
) -> float:
    """
    Compute average nDCG@K.
    nDCG evaluates ranked retrieval quality using binary or graded relevance labels.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevance_by_query:
        return 0.0

    ndcg_scores: list[float] = []

    for result in results:
        relevance_map = relevance_by_query.get(result.query_id, {})

        if not relevance_map:
            ndcg_scores.append(0.0)
            continue

        retrieved_relevances = [
            relevance_map.get(item_id, 0.0)
            for item_id in result.retrieved_ids[:k]
        ]

        dcg = _dcg(retrieved_relevances)
        ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = _dcg(ideal_relevances)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def _dcg(relevances: list[float]) -> float:
    """Compute Discounted Cumulative Gain for a ranked relevance list."""
    score = 0.0
    for index, relevance in enumerate(relevances, start=1):
        score += (2**relevance - 1) / math.log2(index + 1)
    return score


# =============================================================================
# BASEMET ADAPTERS — wraps your functions for CLI + registry dispatch
# =============================================================================

class RecallAtKMetric(BaseMetric):
    """
    BaseMetric adapter for recall_at_k().
    Bridges your function to the eval-engine plugin interface.

    Note: This metric uses RetrievalCase/RetrievalResult typed inputs,
    not the generic question/contexts/answer interface. The score() method
    accepts those for interface compatibility but the real dispatch goes
    through score_from_cases().
    """

    name = "recall_at_k"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Single-query binary recall proxy.
        For full batch recall, use score_from_cases().
        """
        self.validate_inputs(question, contexts, answer)
        if not ground_truth:
            return self.error_result("recall_at_k requires ground_truth for single-query mode")

        gt_lower = ground_truth.lower()
        top_k = contexts[:self.k]
        hit = any(gt_lower in ctx.lower() for ctx in top_k)

        return MetricResult(
            metric_name=self.name,
            score=1.0 if hit else 0.0,
            raw={"k": self.k, "hit": hit, "mode": "single_query_proxy"},
            metadata={"note": "Single-query proxy. Use score_from_cases() for benchmark evaluation."},
        )

    def score_from_cases(
        self,
        cases: list[RetrievalCase],
        results: list[RetrievalResult],
    ) -> MetricResult:
        """Full benchmark Recall@K using your original typed function."""
        score = recall_at_k(cases, results, self.k)
        return MetricResult(
            metric_name=self.name,
            score=score,
            raw={"k": self.k, "num_cases": len(cases), "num_results": len(results)},
            metadata={"mode": "benchmark", "definition": "query_level_binary"},
        )


class MRRMetric(BaseMetric):
    """BaseMetric adapter for mean_reciprocal_rank()."""

    name = "mean_reciprocal_rank"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        self.validate_inputs(question, contexts, answer)
        if not ground_truth:
            return self.error_result("mrr requires ground_truth for single-query mode")

        gt_lower = ground_truth.lower()
        rank_score = 0.0
        for i, ctx in enumerate(contexts, start=1):
            if gt_lower in ctx.lower():
                rank_score = 1.0 / i
                break

        return MetricResult(
            metric_name=self.name,
            score=rank_score,
            raw={"reciprocal_rank": rank_score, "mode": "single_query_proxy"},
        )

    def score_from_cases(
        self,
        cases: list[RetrievalCase],
        results: list[RetrievalResult],
    ) -> MetricResult:
        """Full benchmark MRR using your original typed function."""
        score = mean_reciprocal_rank(cases, results)
        return MetricResult(
            metric_name=self.name,
            score=score,
            raw={"num_cases": len(cases), "num_results": len(results)},
            metadata={"mode": "benchmark"},
        )


class NDCGMetric(BaseMetric):
    """
    BaseMetric adapter for ndcg_at_k().

    Uses exponential DCG: (2^rel - 1) / log2(i+1)
    Requires graded relevance labels (relevance_by_query).
    See Consolidation Eval Suite methodology section for formula declaration.
    """

    name = "ndcg"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        relevance_labels: list[float] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        self.validate_inputs(question, contexts, answer)

        if relevance_labels is not None:
            if len(relevance_labels) != len(contexts):
                return self.error_result(
                    f"relevance_labels length {len(relevance_labels)} != contexts length {len(contexts)}"
                )
            rels = relevance_labels
        elif ground_truth:
            gt_lower = ground_truth.lower()
            rels = [1.0 if gt_lower in ctx.lower() else 0.0 for ctx in contexts]
        else:
            return self.error_result("ndcg requires relevance_labels or ground_truth")

        dcg_val = _dcg(rels[:self.k])
        ideal = _dcg(sorted(rels, reverse=True)[:self.k])
        score = dcg_val / ideal if ideal > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            raw={
                "k": self.k,
                "dcg": dcg_val,
                "idcg": ideal,
                "relevances": rels[:self.k],
                "formula": "exponential: (2^rel-1)/log2(i+1)",
            },
        )

    def score_from_relevance_map(
        self,
        relevance_by_query: Mapping[str, Mapping[str, float]],
        results: list[RetrievalResult],
    ) -> MetricResult:
        """Full benchmark NDCG@K using your original typed function."""
        score = ndcg_at_k(relevance_by_query, results, self.k)
        return MetricResult(
            metric_name=self.name,
            score=score,
            raw={
                "k": self.k,
                "num_queries": len(results),
                "formula": "exponential: (2^rel-1)/log2(i+1)",
            },
            metadata={"mode": "benchmark_graded"},
        )
