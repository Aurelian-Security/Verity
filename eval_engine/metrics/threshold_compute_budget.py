"""
eval_engine/metrics/threshold_compute_budget.py

Consolidation RAG Evaluation — Test 3: Threshold Compute Budget Test

PURPOSE:
    Measures retrieval quality as a function of compute budget —
    specifically, how Consolidation RAG's performance degrades or improves
    as the consolidation cycle's compute allocation changes.

    Consolidation RAG uses a threshold-gated consolidation mechanism:
    the system commits a consolidation cycle only when a compute
    budget threshold is met. This test answers:

        "What is the minimum compute budget required for Consolidation RAG
         to outperform the pre-consolidation baseline?"

    And conversely:
        "Does throwing more compute at consolidation keep improving
         quality, or does it plateau?"

DESIGN:
    - Run retrieval at N compute budget levels (e.g. 25%, 50%, 75%, 100%)
    - Score NDCG@10 at each level
    - Fit a curve to identify the inflection point (minimum budget for
      meaningful improvement over baseline)
    - Report: threshold level, score at threshold, plateau score

HARDWARE NOTE:
    On your RTX 3060 (12GB VRAM) with 96GB RAM, compute budget maps to:
    - Consolidation cycle iteration count
    - Or: number of graph refinement passes
    - Or: max_tokens budget per consolidation LLM call
    Confirm the mapping with Reza before running this test.

OUTPUT:
    BudgetCurveResult — score at each budget level + inflection point.
    Used in Consolidation Eval Suite to justify Consolidation RAG's threshold-gating design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.metrics.retrieval_metrics import ndcg_at_k
from eval_engine.schemas import RetrievalCase, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BudgetPoint:
    """Score at a single compute budget level."""
    budget_level: float          # 0.0–1.0 (fraction of max budget)
    budget_label: str            # Human-readable (e.g. "25%", "512 tokens")
    ndcg_score: float
    delta_from_baseline: float   # Score - baseline (pre-consolidation) score
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def improves_over_baseline(self) -> bool:
        return self.delta_from_baseline > 0.0


@dataclass
class BudgetCurveResult:
    """Full compute budget curve for a single experiment."""
    experiment_id: str
    baseline_score: float              # Pre-consolidation NDCG@K
    points: list[BudgetPoint]          # Scored at each budget level
    inflection_budget: float | None    # Budget level where curve flattens
    plateau_score: float | None        # Score at plateau
    k: int = 10

    @property
    def min_effective_budget(self) -> float | None:
        """
        Minimum budget level where score exceeds baseline by > 5%.
        Returns None if no budget level achieves this.
        """
        threshold = self.baseline_score * 1.05
        for point in sorted(self.points, key=lambda p: p.budget_level):
            if point.ndcg_score >= threshold:
                return point.budget_level
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "baseline_score": self.baseline_score,
            "k": self.k,
            "inflection_budget": self.inflection_budget,
            "plateau_score": self.plateau_score,
            "min_effective_budget": self.min_effective_budget,
            "points": [
                {
                    "budget_level": p.budget_level,
                    "budget_label": p.budget_label,
                    "ndcg_score": p.ndcg_score,
                    "delta_from_baseline": p.delta_from_baseline,
                    "improves_over_baseline": p.improves_over_baseline,
                }
                for p in self.points
            ],
        }

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  COMPUTE BUDGET CURVE: {self.experiment_id}")
        print(f"  Baseline (pre-consolidation): {self.baseline_score:.4f}")
        print(f"  Min effective budget: {self.min_effective_budget}")
        print(f"  Plateau score: {self.plateau_score}")
        print(f"{'='*55}")
        for p in self.points:
            marker = " ✓" if p.improves_over_baseline else ""
            print(
                f"  Budget {p.budget_label:>6}: "
                f"NDCG@{self.k}={p.ndcg_score:.4f} "
                f"(delta={p.delta_from_baseline:+.4f}){marker}"
            )
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Inflection point detection
# ---------------------------------------------------------------------------

def _find_inflection_point(
    budget_levels: list[float],
    scores: list[float],
    plateau_threshold: float = 0.01,
) -> tuple[float | None, float | None]:
    """
    Find where the score curve flattens.

    Returns (inflection_budget, plateau_score).
    Inflection point = first budget level where score increase
    drops below plateau_threshold compared to previous level.
    """
    if len(scores) < 2:
        return None, None

    for i in range(1, len(scores)):
        improvement = scores[i] - scores[i - 1]
        if improvement < plateau_threshold:
            return budget_levels[i], scores[i]

    # No plateau found — curve still rising at max budget
    return budget_levels[-1], scores[-1]


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class ThresholdComputeBudgetMetric(BaseMetric):
    """
    Consolidation RAG Evaluation Test 3: Threshold Compute Budget Test.

    Measures the NDCG@K vs. compute budget curve.
    Identifies the inflection point where additional compute stops
    improving retrieval quality.

    The caller provides a retriever function that accepts a budget_level
    parameter (0.0–1.0) controlling how much consolidation compute is used.

    Usage:
        def retrieve_at_budget(query: str, budget_level: float) -> list[str]:
            return consolidation_rag.retrieve(query, compute_budget=budget_level)

        metric = ThresholdComputeBudgetMetric(k=10)
        curve = metric.run_budget_sweep(
            cases=eval_cases,
            baseline_retriever=retrieve_pre_consolidation,
            budget_retriever=retrieve_at_budget,
            budget_levels=[0.25, 0.50, 0.75, 1.00],
            relevance_by_query=relevance_labels,
        )
    """

    name = "threshold_compute_budget"

    def __init__(self, k: int = 10, plateau_threshold: float = 0.01) -> None:
        self.k = k
        self.plateau_threshold = plateau_threshold

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        budget_level: float = 1.0,
        relevance_labels: list[float] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Single-query score at a given budget level. Delegates to NDCG."""
        from eval_engine.metrics.ndcg import NDCGMetric
        ndcg = NDCGMetric(k=self.k)
        result = ndcg.score(question, contexts, answer, ground_truth, relevance_labels)
        result.metadata["budget_level"] = budget_level
        return result

    def run_budget_sweep(
        self,
        cases: list[RetrievalCase],
        baseline_retriever: Callable[[str], list[str]],
        budget_retriever: Callable[[str, float], list[str]],
        budget_levels: list[float],
        relevance_by_query: dict[str, dict[str, float]],
        budget_labels: list[str] | None = None,
        experiment_id: str = "budget_sweep",
    ) -> BudgetCurveResult:
        """
        Run NDCG@K at each budget level and compute the curve.

        Args:
            cases:               Benchmark cases
            baseline_retriever:  Retriever fn with NO consolidation (budget=0)
            budget_retriever:    Retriever fn accepting (query, budget_level)
            budget_levels:       List of budget fractions [0.0–1.0]
            relevance_by_query:  Graded relevance labels
            budget_labels:       Human-readable labels (defaults to pct strings)
            experiment_id:       Run identifier

        Returns:
            BudgetCurveResult with full curve and inflection point
        """
        # Validate budget levels
        if not all(0.0 <= b <= 1.0 for b in budget_levels):
            raise ValueError("All budget_levels must be between 0.0 and 1.0")
        budget_levels = sorted(budget_levels)

        if budget_labels is None:
            budget_labels = [f"{int(b*100)}%" for b in budget_levels]

        # Baseline score (pre-consolidation)
        baseline_results = self._run_retriever(cases, lambda q: baseline_retriever(q))
        baseline_score = ndcg_at_k(relevance_by_query, baseline_results, self.k)
        logger.info(f"[budget_sweep] Baseline NDCG@{self.k}: {baseline_score:.4f}")

        # Score at each budget level
        points = []
        scores_for_inflection = []

        for level, label in zip(budget_levels, budget_labels):
            results = self._run_retriever(
                cases, lambda q, lv=level: budget_retriever(q, lv)
            )
            score = ndcg_at_k(relevance_by_query, results, self.k)
            delta = score - baseline_score

            point = BudgetPoint(
                budget_level=level,
                budget_label=label,
                ndcg_score=score,
                delta_from_baseline=delta,
            )
            points.append(point)
            scores_for_inflection.append(score)
            logger.info(f"[budget_sweep] Budget {label}: NDCG@{self.k}={score:.4f}, delta={delta:+.4f}")

        inflection_budget, plateau_score = _find_inflection_point(
            budget_levels, scores_for_inflection, self.plateau_threshold
        )

        curve = BudgetCurveResult(
            experiment_id=experiment_id,
            baseline_score=baseline_score,
            points=points,
            inflection_budget=inflection_budget,
            plateau_score=plateau_score,
            k=self.k,
        )
        curve.print_summary()
        return curve

    def _run_retriever(
        self,
        cases: list[RetrievalCase],
        retriever_fn: Callable[[str], list[str]],
    ) -> list[RetrievalResult]:
        results = []
        for case in cases:
            try:
                retrieved = retriever_fn(case.query)
                results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=retrieved))
            except Exception as e:
                logger.error(f"Retriever failed for '{case.query_id}': {e}")
                results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=[]))
        return results
