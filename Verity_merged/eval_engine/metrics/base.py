"""
eval_engine/metrics/base.py

Abstract base class for all eval-engine metric plugins.

Every metric — whether RAGAS grounding, NDCG, LlamaGuard safety,
or a custom plugin — must implement this interface. The MetricsRegistry
uses this contract to discover and dispatch metrics at runtime.

Consolidation RAG Evaluation metrics (ragas_consolidation_delta, per_stage_ablation,
threshold_compute_budget, single_session_poisoning, query_perturbation)
each have a corresponding subclass in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """
    Standardized output from a single metric evaluation.

    Every metric returns one of these. The EvalRunner aggregates
    results across queries and passes them to ReportGenerator.
    """
    metric_name: str
    score: float
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        status = "OK" if self.succeeded else f"ERR: {self.error}"
        return f"MetricResult({self.metric_name}={self.score:.4f} [{status}])"


@dataclass
class ConsolidationDeltaResult(MetricResult):
    """
    Extended result for ragas_consolidation_delta.
    Captures pre/post scores separately so delta is auditable.
    """
    pre_score: float = 0.0
    post_score: float = 0.0

    @property
    def delta(self) -> float:
        return self.post_score - self.pre_score


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseMetric(ABC):
    """
    Abstract base for all eval-engine metric plugins.

    Subclasses implement:
        - name (property): unique string identifier matching TestName enum
        - score(): sync single-query evaluation
        - score_async(): async single-query evaluation (default wraps sync)
        - score_batch(): async batch evaluation (default maps score_async)

    Plugin registration:
        Custom metrics register themselves by implementing this class
        and pointing the EvalConfig metric.kwargs['plugin_path'] at
        the module. The MetricsRegistry loads them dynamically.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier. Must match a TestName enum value."""
        ...

    @abstractmethod
    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Synchronous single-query scoring.

        Args:
            question:     The query posed to the RAG system.
            contexts:     Retrieved context chunks passed to the generator.
            answer:       The RAG system's generated answer.
            ground_truth: Reference answer for metrics that require it (NDCG, delta).
            **kwargs:     Metric-specific parameters from MetricConfig.kwargs.

        Returns:
            MetricResult with score in [0, 1] (higher = better) unless
            the metric has a different natural range (e.g. NDCG is [0,1],
            cost is unbounded — document exceptions in subclass docstring).
        """
        ...

    async def score_async(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Async single-query scoring. Default wraps sync score().
        Override for metrics that make async LLM calls natively.
        """
        return self.score(question, contexts, answer, ground_truth, **kwargs)

    async def score_batch(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Async batch scoring. Default maps score_async over records.
        Override for metrics that support native batch APIs (e.g. RAGAS dataset eval).

        Each record must have keys: question, contexts, answer, ground_truth (optional).
        """
        results = []
        for rec in records:
            result = await self.score_async(
                question=rec["question"],
                contexts=rec.get("contexts", []),
                answer=rec["answer"],
                ground_truth=rec.get("ground_truth"),
                **kwargs,
            )
            results.append(result)
        return results

    def validate_inputs(
        self,
        question: str,
        contexts: list[str],
        answer: str,
    ) -> None:
        """
        Basic input validation. Call at the top of score() in subclasses.
        Raises ValueError on empty required fields.
        """
        if not question or not question.strip():
            raise ValueError(f"[{self.name}] question must not be empty")
        if not answer or not answer.strip():
            raise ValueError(f"[{self.name}] answer must not be empty")
        if not isinstance(contexts, list):
            raise ValueError(f"[{self.name}] contexts must be a list of strings")

    def error_result(self, error: str) -> MetricResult:
        """Convenience: return a failed MetricResult without raising."""
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            error=error,
        )
