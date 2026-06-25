"""
eval_engine/metrics/ragas_consolidation_delta.py

Consolidation RAG Evaluation — Test 1: RAGAS Pre/Post Consolidation Delta

SOURCE: Scaffold (new code), REFACTORED from original scaffold design.
CHANGE: Now calls run_ragas_evaluation() from ragas_runner.py instead of
        duplicating RAGAS evaluation logic.

Measures the change in RAGAS faithfulness, context_recall, and context_precision
scores before and after Consolidation RAG's offline consolidation cycle.

Primary metric: NDCG@10 (retrieval_metrics.py)
This module: RAGAS component scores + delta computation on top of ragas_runner

RAGAS backend must be configured before use:
    from eval_engine.metrics.ragas_runner import configure_ragas_llm
    configure_ragas_llm(model="claude-sonnet-4-6", provider="anthropic")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.metrics.ragas_adapter import RagasCase
from eval_engine.metrics.ragas_runner import run_ragas_evaluation

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationDeltaResult(MetricResult):
    """
    Extended MetricResult for pre/post consolidation delta.
    score (inherited) = faithfulness_delta (primary scalar for Wilcoxon).
    """
    pre_faithfulness: float = 0.0
    post_faithfulness: float = 0.0
    pre_context_recall: float = 0.0
    post_context_recall: float = 0.0
    pre_context_precision: float = 0.0
    post_context_precision: float = 0.0

    @property
    def faithfulness_delta(self) -> float:
        return self.post_faithfulness - self.pre_faithfulness

    @property
    def context_recall_delta(self) -> float:
        return self.post_context_recall - self.pre_context_recall

    @property
    def context_precision_delta(self) -> float:
        return self.post_context_precision - self.pre_context_precision

    def summary(self) -> dict[str, float]:
        return {
            "faithfulness_delta": self.faithfulness_delta,
            "context_recall_delta": self.context_recall_delta,
            "context_precision_delta": self.context_precision_delta,
            "pre_faithfulness": self.pre_faithfulness,
            "post_faithfulness": self.post_faithfulness,
            "pre_context_recall": self.pre_context_recall,
            "post_context_recall": self.post_context_recall,
            "pre_context_precision": self.pre_context_precision,
            "post_context_precision": self.post_context_precision,
        }


class RAGASConsolidationDelta(BaseMetric):
    """
    RAGAS pre/post consolidation delta metric.
    Wraps run_ragas_evaluation() — does not duplicate RAGAS logic.
    Calls the runner twice (pre phase, post phase) and computes deltas.
    Primary use: Consolidation RAG Evaluation Test 1.
    """

    name = "ragas_consolidation_delta"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        pre_contexts: list[str] | None = None,
        pre_answer: str | None = None,
        **kwargs: Any,
    ) -> ConsolidationDeltaResult:
        self.validate_inputs(question, contexts, answer)

        post_case = RagasCase(question=question, answer=answer, contexts=contexts, ground_truth=ground_truth)
        post_scores = run_ragas_evaluation([post_case], return_partial=True)

        if pre_contexts is not None and pre_answer is not None:
            pre_case = RagasCase(question=question, answer=pre_answer, contexts=pre_contexts, ground_truth=ground_truth)
            pre_scores = run_ragas_evaluation([pre_case], return_partial=True)
        else:
            logger.warning(
                f"[{self.name}] No pre_contexts/pre_answer for '{question[:60]}'. Delta=0."
            )
            pre_scores = {"faithfulness": 0.0, "context_recall": 0.0, "context_precision": 0.0}

        pre_faith = float(pre_scores.get("faithfulness", 0.0))
        post_faith = float(post_scores.get("faithfulness", 0.0))

        result = ConsolidationDeltaResult(
            metric_name=self.name,
            score=post_faith - pre_faith,
            pre_faithfulness=pre_faith,
            post_faithfulness=post_faith,
            pre_context_recall=float(pre_scores.get("context_recall", 0.0)),
            post_context_recall=float(post_scores.get("context_recall", 0.0)),
            pre_context_precision=float(pre_scores.get("context_precision", 0.0)),
            post_context_precision=float(post_scores.get("context_precision", 0.0)),
            raw={
                "pre_ragas": {k: v for k, v in pre_scores.items() if k != "_meta"},
                "post_ragas": {k: v for k, v in post_scores.items() if k != "_meta"},
            },
            metadata={"has_ground_truth": ground_truth is not None, "has_pre_phase": pre_contexts is not None},
        )
        logger.info(f"[{self.name}] faithfulness delta={result.faithfulness_delta:+.4f}")
        return result

    async def score_async(self, question, contexts, answer, ground_truth=None, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.score(question, contexts, answer, ground_truth, **kwargs))

    def score_batch_from_cases(
        self,
        pre_cases: list[RagasCase],
        post_cases: list[RagasCase],
        batch_size: int = 50,
    ) -> list[ConsolidationDeltaResult]:
        from eval_engine.metrics.ragas_runner import run_ragas_evaluation_batch

        if len(pre_cases) != len(post_cases):
            raise ValueError(f"pre_cases ({len(pre_cases)}) != post_cases ({len(post_cases)})")

        pre_batches = run_ragas_evaluation_batch(pre_cases, batch_size=batch_size)
        post_batches = run_ragas_evaluation_batch(post_cases, batch_size=batch_size)

        results = []
        for i, (pre, post) in enumerate(zip(pre_batches, post_batches)):
            pre_faith = float(pre.get("faithfulness", 0.0))
            post_faith = float(post.get("faithfulness", 0.0))
            results.append(ConsolidationDeltaResult(
                metric_name=self.name,
                score=post_faith - pre_faith,
                pre_faithfulness=pre_faith,
                post_faithfulness=post_faith,
                pre_context_recall=float(pre.get("context_recall", 0.0)),
                post_context_recall=float(post.get("context_recall", 0.0)),
                pre_context_precision=float(pre.get("context_precision", 0.0)),
                post_context_precision=float(post.get("context_precision", 0.0)),
                raw={"pre_ragas": pre, "post_ragas": post},
                metadata={"batch_index": i},
            ))
        return results
