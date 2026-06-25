"""
eval_engine/metrics/ragas_grounding.py

RAGAS Retrieval Grounding Metric — BaseMetric adapter.

PURPOSE:
    Wraps run_ragas_evaluation() as a BaseMetric plugin so it can be
    invoked via CLI and registered in MetricsRegistry.

    This is distinct from ragas_consolidation_delta.py:
        - ragas_grounding.py: single-phase grounding score (no delta)
        - ragas_consolidation_delta.py: pre/post delta computation

    Use ragas_grounding for:
        - rag-eval-harness baseline grounding scores
        - Single-phase Consolidation RAG grounding (post-consolidation only)
        - Any eval where you want context_recall, context_precision,
          faithfulness without a comparison phase

RAGAS backend must be configured before use:
    from eval_engine.metrics.ragas_runner import configure_ragas_llm
    configure_ragas_llm(model="claude-sonnet-4-6", provider="anthropic")
"""

from __future__ import annotations

import logging
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.metrics.ragas_adapter import RagasCase
from eval_engine.metrics.ragas_runner import run_ragas_evaluation

logger = logging.getLogger(__name__)


class RAGASGroundingMetric(BaseMetric):
    """
    Single-phase RAGAS grounding evaluation.

    Returns faithfulness as the primary scalar score.
    Full context_recall, context_precision, faithfulness available in result.raw.

    Score: faithfulness (primary — measures factual consistency of answer
           with respect to retrieved contexts).
    """

    name = "ragas_grounding"

    def __init__(self, primary_metric: str = "faithfulness") -> None:
        """
        Args:
            primary_metric: Which RAGAS metric to use as the scalar score.
                            One of: "faithfulness", "context_recall", "context_precision"
        """
        valid = {"faithfulness", "context_recall", "context_precision"}
        if primary_metric not in valid:
            raise ValueError(f"primary_metric must be one of {valid}, got '{primary_metric}'")
        self.primary_metric = primary_metric

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Run RAGAS grounding evaluation for a single query.

        Returns:
            MetricResult with score = primary_metric value.
            result.raw contains all three RAGAS scores + _meta block.
        """
        self.validate_inputs(question, contexts, answer)

        case = RagasCase(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        scores = run_ragas_evaluation([case], return_partial=True)
        primary_score = float(scores.get(self.primary_metric, 0.0))
        success = scores.get("_meta", {}).get("success", False)

        if not success:
            error_msg = scores.get("_meta", {}).get("error", "RAGAS evaluation failed")
            logger.warning(f"[ragas_grounding] Partial result returned: {error_msg}")

        return MetricResult(
            metric_name=self.name,
            score=primary_score,
            raw={k: v for k, v in scores.items() if k != "_meta"},
            metadata={
                "primary_metric": self.primary_metric,
                "ragas_meta": scores.get("_meta", {}),
                "has_ground_truth": ground_truth is not None,
            },
            error=None if success else scores.get("_meta", {}).get("error"),
        )

    async def score_async(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Async wrapper — RAGAS is synchronous; runs in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.score(question, contexts, answer, ground_truth, **kwargs),
        )
