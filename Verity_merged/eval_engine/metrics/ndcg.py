"""
eval_engine/metrics/ndcg.py

NDCG@K (Normalized Discounted Cumulative Gain at K)

Primary retrieval quality metric for consolidation-based RAG evaluation.
Also used in rag-eval-harness harness as post-hoc retrieval scoring.

NDCG@10 is the pre-registered primary metric for the consolidation
delta evaluation — measures ranking quality of post-consolidation
retrieved documents against ground-truth relevance labels.

Score range: [0, 1] — higher is better.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


def _dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute DCG@K given a list of relevance scores in ranked order.
    Uses log base 2: DCG = sum(rel_i / log2(i+2)) for i in [0, k)
    """
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    """
    Compute NDCG@K.

    Args:
        relevances: relevance scores in the order they were retrieved
        k:          cutoff rank

    Returns:
        float in [0, 1]
    """
    if not relevances:
        return 0.0

    dcg = _dcg_at_k(relevances, k)
    # Ideal DCG: sort relevances descending
    ideal = _dcg_at_k(sorted(relevances, reverse=True), k)

    if ideal == 0.0:
        return 0.0

    return dcg / ideal


class NDCGMetric(BaseMetric):
    """
    NDCG@K retrieval evaluation metric.

    Two modes:
      1. Label mode: caller provides explicit relevance scores per context chunk
         (use when you have annotated relevance judgments)

      2. Binary mode: relevance inferred by checking if each context chunk
         contains the ground truth answer string (simple proxy)

    Consolidation RAG Evaluation uses label mode with nugget-based evaluation
    as secondary metric (see nugget_eval.py — Phase 2 addition).

    rag-eval-harness uses binary mode against the eval set ground truth.
    """

    name = "ndcg"

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
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
        """
        Compute NDCG@K for a single query.

        Args:
            question:         Query string (used for logging)
            contexts:         Retrieved context chunks in ranked order
            answer:           Generated answer (not used in NDCG computation)
            ground_truth:     Reference text for binary relevance inference
            relevance_labels: Explicit relevance scores [0..1] per context chunk.
                              If provided, used directly (label mode).
                              If None, binary mode inferred from ground_truth.

        Returns:
            MetricResult with score = NDCG@K
        """
        self.validate_inputs(question, contexts, answer)

        if not contexts:
            return self.error_result("contexts list is empty — cannot compute NDCG")

        if relevance_labels is not None:
            if len(relevance_labels) != len(contexts):
                return self.error_result(
                    f"relevance_labels length ({len(relevance_labels)}) "
                    f"!= contexts length ({len(contexts)})"
                )
            rels = relevance_labels
            mode = "label"
        elif ground_truth:
            # Binary mode: 1.0 if ground truth string appears in chunk, else 0.0
            gt_lower = ground_truth.lower()
            rels = [
                1.0 if gt_lower in ctx.lower() else 0.0
                for ctx in contexts
            ]
            mode = "binary"
        else:
            return self.error_result(
                "NDCG requires either relevance_labels or ground_truth"
            )

        ndcg_score = _ndcg_at_k(rels, self.k)

        return MetricResult(
            metric_name=self.name,
            score=ndcg_score,
            raw={
                "relevances": rels,
                "k": self.k,
                "dcg": _dcg_at_k(rels, self.k),
                "ideal_dcg": _dcg_at_k(sorted(rels, reverse=True), self.k),
            },
            metadata={
                "mode": mode,
                "num_contexts": len(contexts),
                "question_preview": question[:80],
            },
        )
