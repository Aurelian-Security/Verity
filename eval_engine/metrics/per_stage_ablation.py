"""
eval_engine/metrics/per_stage_ablation.py

Consolidation RAG Evaluation — Test 2: Per-Stage Ablation

PURPOSE:
    Measures the contribution of each individual stage in Consolidation RAG's
    consolidation pipeline to overall retrieval quality.

    Ablation = disabling one stage at a time and measuring score drop.
    A large drop when stage X is disabled means stage X is doing real work.
    A negligible drop means the stage is redundant — important finding for
    architecture efficiency claims in Consolidation Eval Suite.

DREAMRAG CONSOLIDATION STAGES (expected — confirm with Reza):
    Stage 0: Baseline (no consolidation)
    Stage 1: Graph ingestion / initial embedding
    Stage 2: Consolidation-phase edge pruning (graph downscaling)
    Stage 3: REM-phase pattern synthesis / knowledge graph refinement
    Stage 4: Threshold-gated consolidation commit
    Stage 5: Full pipeline (all stages active)

ABLATION DESIGN:
    For each stage S:
        - Run retrieval with all stages EXCEPT S active
        - Score with NDCG@10 + RAGAS faithfulness
        - Delta from full-pipeline score = contribution of stage S

    Largest delta = most critical stage.
    Near-zero delta = candidate for pruning.

OUTPUT:
    AblationResult per stage — score, delta from full pipeline, rank.
    Aggregated into AblationReport for Consolidation Eval Suite results table.

PAPER 1 NOTE:
    This test does NOT require LLM judge calls if using NDCG only.
    Add RAGAS grounding for richer per-stage analysis (adds cost).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.metrics.retrieval_metrics import ndcg_at_k, NDCGMetric
from eval_engine.schemas import RetrievalCase, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage result container
# ---------------------------------------------------------------------------

@dataclass
class AblationStageResult:
    """Result for a single ablated stage."""
    stage_id: int
    stage_name: str
    score: float                        # NDCG@K (or primary metric) with this stage ablated
    full_pipeline_score: float          # Score with all stages active
    delta: float                        # full_pipeline_score - score (positive = stage contributed)
    metric_name: str = "ndcg"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contribution_pct(self) -> float:
        """Stage contribution as % of full pipeline score."""
        if self.full_pipeline_score == 0:
            return 0.0
        return (self.delta / self.full_pipeline_score) * 100

    @property
    def is_critical(self) -> bool:
        """True if ablating this stage causes > 10% score drop."""
        return self.contribution_pct > 10.0


@dataclass
class AblationReport:
    """Aggregated per-stage ablation results."""
    experiment_id: str
    full_pipeline_score: float
    stage_results: list[AblationStageResult]

    @property
    def critical_stages(self) -> list[AblationStageResult]:
        return [s for s in self.stage_results if s.is_critical]

    @property
    def redundant_stages(self) -> list[AblationStageResult]:
        return [s for s in self.stage_results if not s.is_critical]

    def ranked(self) -> list[AblationStageResult]:
        """Stages ranked by contribution (most critical first)."""
        return sorted(self.stage_results, key=lambda s: s.delta, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "full_pipeline_score": self.full_pipeline_score,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "stage_name": s.stage_name,
                    "score_ablated": s.score,
                    "delta": s.delta,
                    "contribution_pct": round(s.contribution_pct, 2),
                    "is_critical": s.is_critical,
                }
                for s in self.ranked()
            ],
        }

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  ABLATION REPORT: {self.experiment_id}")
        print(f"  Full pipeline score: {self.full_pipeline_score:.4f}")
        print(f"{'='*55}")
        for s in self.ranked():
            marker = " *** CRITICAL" if s.is_critical else ""
            print(
                f"  Stage {s.stage_id} ({s.stage_name}): "
                f"score={s.score:.4f}, delta={s.delta:+.4f}, "
                f"contribution={s.contribution_pct:.1f}%{marker}"
            )
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Ablation metric
# ---------------------------------------------------------------------------

class PerStageAblationMetric(BaseMetric):
    """
    Consolidation RAG Evaluation Test 2: Per-Stage Ablation.

    Requires the caller to provide a retrieval function per stage —
    a callable that runs retrieval with a specific stage disabled.

    This design is intentional: the eval engine doesn't control
    Consolidation RAG's pipeline internals. The caller wires up the ablation
    by providing stage-specific retrieval functions.

    Usage:
        def retrieve_without_stage_2(query: str) -> list[str]:
            # Run with consolidation stage disabled
            return consolidation_rag.retrieve(query, disable_stages=[2])

        metric = PerStageAblationMetric(k=10)
        report = metric.run_ablation(
            cases=eval_cases,
            stage_retrievers={
                0: ("baseline", retrieve_no_consolidation),
                2: ("nrem_pruning", retrieve_without_stage_2),
                3: ("rem_synthesis", retrieve_without_stage_3),
            },
            full_pipeline_retriever=retrieve_full_pipeline,
            relevance_by_query=relevance_labels,
        )
    """

    name = "per_stage_ablation"

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self._ndcg = NDCGMetric(k=k)

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
        Score a single query for the ablation metric.
        Delegates to NDCGMetric — ablation logic is in run_ablation().
        """
        return self._ndcg.score(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
            relevance_labels=relevance_labels,
        )

    def run_ablation(
        self,
        cases: list[RetrievalCase],
        stage_retrievers: dict[int, tuple[str, Callable[[str], list[str]]]],
        full_pipeline_retriever: Callable[[str], list[str]],
        relevance_by_query: dict[str, dict[str, float]],
        experiment_id: str = "ablation",
    ) -> AblationReport:
        """
        Run full per-stage ablation across all cases.

        Args:
            cases:                  Benchmark cases (RetrievalCase list)
            stage_retrievers:       {stage_id: (stage_name, retriever_fn)}
                                    Each fn: (query: str) -> list[retrieved_ids]
            full_pipeline_retriever: Retriever with all stages active
            relevance_by_query:     {query_id: {doc_id: relevance_score}}
            experiment_id:          Run identifier for the report

        Returns:
            AblationReport with per-stage scores and deltas
        """
        # Score full pipeline
        full_results = self._run_retriever(cases, full_pipeline_retriever)
        full_score = ndcg_at_k(relevance_by_query, full_results, self.k)
        logger.info(f"[ablation] Full pipeline NDCG@{self.k}: {full_score:.4f}")

        stage_results = []
        for stage_id, (stage_name, retriever_fn) in stage_retrievers.items():
            ablated_results = self._run_retriever(cases, retriever_fn)
            ablated_score = ndcg_at_k(relevance_by_query, ablated_results, self.k)
            delta = full_score - ablated_score

            result = AblationStageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                score=ablated_score,
                full_pipeline_score=full_score,
                delta=delta,
                metric_name=f"ndcg@{self.k}",
            )
            stage_results.append(result)
            logger.info(
                f"[ablation] Stage {stage_id} ({stage_name}): "
                f"score={ablated_score:.4f}, delta={delta:+.4f}, "
                f"contribution={result.contribution_pct:.1f}%"
            )

        report = AblationReport(
            experiment_id=experiment_id,
            full_pipeline_score=full_score,
            stage_results=stage_results,
        )
        report.print_summary()
        return report

    def _run_retriever(
        self,
        cases: list[RetrievalCase],
        retriever_fn: Callable[[str], list[str]],
    ) -> list[RetrievalResult]:
        """Run retriever function over all cases, return RetrievalResult list."""
        results = []
        for case in cases:
            try:
                retrieved_ids = retriever_fn(case.query)
                results.append(RetrievalResult(
                    query_id=case.query_id,
                    retrieved_ids=retrieved_ids,
                ))
            except Exception as e:
                logger.error(f"Retriever failed for query '{case.query_id}': {e}")
                results.append(RetrievalResult(
                    query_id=case.query_id,
                    retrieved_ids=[],
                ))
        return results
