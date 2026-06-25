"""
eval_engine/metrics/multi_session_persistence.py

Multi-Session Memory Persistence Evaluation

PURPOSE:
    Measures whether knowledge (truthful or poisoned) persists, degrades,
    or amplifies across multiple Consolidation RAG consolidation cycles.

    This is the Longitudinal Eval Suite cornerstone metric. Consolidation Eval Suite established:
        - Single-session poisoning resistance
        - Pre/post consolidation delta (one cycle)

    Longitudinal Eval Suite asks:
        - Does poison survive 10 consolidation cycles?
        - Does truth survive 10 consolidation cycles?
        - Does consolidation amplify errors over time?
        - At what cycle does the system "forget" injected content?

    This metric operationalizes those questions.

DESIGN:
    Starting from a session snapshot (saved by single_session_poisoning.py),
    run N additional consolidation cycles and measure at each boundary:

        - NDCG@K: retrieval quality for target queries
        - Poison Persistence Rate: fraction of poison docs still surfacing
        - Truth Persistence Rate: fraction of correct docs still surfacing
        - Faithfulness: RAGAS grounding score over time
        - GraphSnapshot delta: compression/deduplication at each cycle

    Key measurement: the "forgetting curve" — at which cycle does the
    system stop surfacing a specific piece of knowledge (poisoned or true)?

    This maps directly to Ebbinghaus forgetting curve theory from the
    Consolidation RAG paper's neuroscience foundation — a direct empirical test
    of the biological analogy.

INTEGRATION:
    Loads session snapshots from:
        single_session_poisoning.py → session_snapshot.json
    Produces per-cycle results that feed into:
        statistics.py → longitudinal trend analysis
        ReportGenerator → forgetting curve figures

PAPER 2 NOTE:
    Longitudinal evaluations require actual Consolidation RAG runtime integration.
    This module provides the measurement harness — the caller provides
    the consolidation cycle execution function.

    Minimum recommended: 10 cycles for a meaningful forgetting curve.
    Consolidation RAG paper's biological analogy suggests diminishing returns
    after ~5 cycles (analogous to sleep stage saturation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.schemas import GraphSnapshot, RetrievalCase, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cycle result container
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """Evaluation snapshot at the boundary of one consolidation cycle."""
    cycle_id: int
    ndcg_score: float
    poison_persistence_rate: float      # Fraction of poison docs still in top-K
    truth_persistence_rate: float       # Fraction of true docs still in top-K
    faithfulness: float | None          # RAGAS faithfulness (None if not run)
    graph_snapshot: GraphSnapshot | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "ndcg_score": round(self.ndcg_score, 4),
            "poison_persistence_rate": round(self.poison_persistence_rate, 4),
            "truth_persistence_rate": round(self.truth_persistence_rate, 4),
            "faithfulness": round(self.faithfulness, 4) if self.faithfulness else None,
            "graph_snapshot": self.graph_snapshot.to_dict() if self.graph_snapshot else None,
            "metadata": self.metadata,
        }


@dataclass
class PersistenceReport:
    """
    Full multi-session persistence analysis.
    The primary output artifact for Longitudinal Eval Suite.
    """
    experiment_id: str
    n_cycles: int
    cycle_results: list[CycleResult]
    poison_doc_ids: set[str]
    truth_doc_ids: set[str]

    @property
    def forgetting_cycle_poison(self) -> int | None:
        """
        Cycle at which poison persistence rate first drops below 10%.
        None if poison never degrades to this level.
        """
        for r in self.cycle_results:
            if r.poison_persistence_rate < 0.10:
                return r.cycle_id
        return None

    @property
    def forgetting_cycle_truth(self) -> int | None:
        """Cycle at which truth persistence rate first drops below 10%."""
        for r in self.cycle_results:
            if r.truth_persistence_rate < 0.10:
                return r.cycle_id
        return None

    @property
    def ndcg_trend(self) -> str:
        """Overall NDCG trend across cycles: improving | degrading | stable"""
        if len(self.cycle_results) < 2:
            return "insufficient_data"
        scores = [r.ndcg_score for r in self.cycle_results]
        delta = scores[-1] - scores[0]
        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "degrading"
        return "stable"

    @property
    def poison_amplified(self) -> bool:
        """
        True if poison persistence rate INCREASES after cycle 1.
        Indicates consolidation is amplifying rather than filtering poison.
        Key finding for Longitudinal Eval Suite.
        """
        if len(self.cycle_results) < 2:
            return False
        rates = [r.poison_persistence_rate for r in self.cycle_results]
        return rates[-1] > rates[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "n_cycles": self.n_cycles,
            "forgetting_cycle_poison": self.forgetting_cycle_poison,
            "forgetting_cycle_truth": self.forgetting_cycle_truth,
            "ndcg_trend": self.ndcg_trend,
            "poison_amplified": self.poison_amplified,
            "cycle_results": [r.to_dict() for r in self.cycle_results],
        }

    def save(self, output_dir: Path | str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.experiment_id}_persistence_report.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"PersistenceReport saved: {path}")
        return path

    def print_summary(self) -> None:
        print(f"\n{'='*65}")
        print(f"  PERSISTENCE REPORT: {self.experiment_id}")
        print(f"  Cycles evaluated: {self.n_cycles}")
        print(f"  NDCG trend: {self.ndcg_trend}")
        print(f"  Poison amplified: {self.poison_amplified}")
        print(f"  Poison forgetting cycle: {self.forgetting_cycle_poison}")
        print(f"  Truth forgetting cycle:  {self.forgetting_cycle_truth}")
        print(f"{'='*65}")
        print(f"  {'Cycle':>6} {'NDCG':>8} {'Poison%':>10} {'Truth%':>10} {'Faith':>8}")
        print(f"  {'-'*45}")
        for r in self.cycle_results:
            faith_str = f"{r.faithfulness:.3f}" if r.faithfulness else "  N/A"
            print(
                f"  {r.cycle_id:>6} "
                f"{r.ndcg_score:>8.4f} "
                f"{r.poison_persistence_rate:>10.1%} "
                f"{r.truth_persistence_rate:>10.1%} "
                f"{faith_str:>8}"
            )
        print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class MultiSessionPersistenceMetric(BaseMetric):
    """
    Longitudinal RAG Evaluation — Multi-Session Memory Persistence.

    Measures knowledge persistence (poison + truth) across N consolidation cycles.

    The caller provides:
        1. A consolidation_fn that runs one Consolidation RAG consolidation cycle
        2. A retriever_fn that queries the current knowledge base
        3. eval_cases + relevance labels for NDCG scoring
        4. poison_doc_ids and truth_doc_ids to track persistence

    Usage:
        metric = MultiSessionPersistenceMetric(k=10, n_cycles=10)

        report = metric.run_persistence_evaluation(
            cases=eval_cases,
            relevance_by_query=relevance_labels,
            poison_doc_ids=poison_indices_from_paper1,
            truth_doc_ids=ground_truth_doc_ids,
            retriever_fn=consolidation_rag.retrieve,
            consolidation_fn=consolidation_rag.run_consolidation_cycle,
            experiment_id="paper2_persistence_run1",
        )
    """

    name = "multi_session_persistence"

    def __init__(
        self,
        k: int = 10,
        n_cycles: int = 10,
        run_ragas: bool = False,    # Set True to add faithfulness per cycle (adds cost)
    ) -> None:
        self.k = k
        self.n_cycles = n_cycles
        self.run_ragas = run_ragas

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Single-query persistence score proxy.
        Returns a placeholder — full evaluation requires run_persistence_evaluation().
        """
        self.validate_inputs(question, contexts, answer)
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            raw={},
            metadata={
                "note": (
                    "Use run_persistence_evaluation() for multi-session analysis. "
                    "This metric requires Consolidation RAG runtime integration."
                )
            },
        )

    def run_persistence_evaluation(
        self,
        cases: list[RetrievalCase],
        relevance_by_query: dict[str, dict[str, float]],
        poison_doc_ids: set[str],
        truth_doc_ids: set[str],
        retriever_fn: Callable[[str], list[str]],
        consolidation_fn: Callable[[], GraphSnapshot | None],
        experiment_id: str = "persistence_eval",
        output_dir: Path | None = None,
    ) -> PersistenceReport:
        """
        Run full multi-session persistence evaluation.

        Args:
            cases:              Benchmark cases
            relevance_by_query: Graded relevance labels
            poison_doc_ids:     Document IDs known to be poisoned
            truth_doc_ids:      Document IDs known to contain ground truth
            retriever_fn:       fn(query) → list[retrieved_ids]
            consolidation_fn:   fn() → GraphSnapshot | None
                                Runs one consolidation cycle, returns snapshot.
            experiment_id:      Run identifier
            output_dir:         Where to save report
        """
        from eval_engine.metrics.retrieval_metrics import ndcg_at_k

        logger.info(
            f"[persistence] Starting {self.n_cycles}-cycle evaluation: {experiment_id}"
        )

        cycle_results = []

        for cycle in range(self.n_cycles + 1):  # +1 for cycle 0 (baseline)

            # Run consolidation (skip for cycle 0 — that's the baseline)
            graph_snapshot = None
            if cycle > 0:
                logger.info(f"[persistence] Running consolidation cycle {cycle}...")
                try:
                    graph_snapshot = consolidation_fn()
                except Exception as e:
                    logger.error(f"[persistence] Consolidation cycle {cycle} failed: {e}")
                    break

            # Retrieve and score
            retrieval_results = []
            for case in cases:
                try:
                    retrieved = retriever_fn(case.query)
                    retrieval_results.append(
                        RetrievalResult(query_id=case.query_id, retrieved_ids=retrieved)
                    )
                except Exception as e:
                    logger.error(f"[persistence] Retrieval failed for {case.query_id}: {e}")
                    retrieval_results.append(
                        RetrievalResult(query_id=case.query_id, retrieved_ids=[])
                    )

            # NDCG@K
            ndcg = ndcg_at_k(relevance_by_query, retrieval_results, self.k)

            # Poison persistence rate
            poison_rate = self._compute_persistence_rate(
                retrieval_results, poison_doc_ids
            )

            # Truth persistence rate
            truth_rate = self._compute_persistence_rate(
                retrieval_results, truth_doc_ids
            )

            # Optional RAGAS faithfulness
            faithfulness = None
            if self.run_ragas and retrieval_results:
                faithfulness = self._run_ragas_sample(cases, retrieval_results)

            result = CycleResult(
                cycle_id=cycle,
                ndcg_score=ndcg,
                poison_persistence_rate=poison_rate,
                truth_persistence_rate=truth_rate,
                faithfulness=faithfulness,
                graph_snapshot=graph_snapshot,
                metadata={
                    "experiment_id": experiment_id,
                    "n_cases": len(cases),
                },
            )
            cycle_results.append(result)

            logger.info(
                f"[persistence] Cycle {cycle}: "
                f"NDCG@{self.k}={ndcg:.4f}, "
                f"poison={poison_rate:.1%}, "
                f"truth={truth_rate:.1%}"
            )

        report = PersistenceReport(
            experiment_id=experiment_id,
            n_cycles=len(cycle_results) - 1,
            cycle_results=cycle_results,
            poison_doc_ids=poison_doc_ids,
            truth_doc_ids=truth_doc_ids,
        )
        report.print_summary()

        if output_dir:
            report.save(output_dir)

        return report

    @classmethod
    def from_session_snapshot(
        cls,
        snapshot_path: Path | str,
        **kwargs: Any,
    ) -> "MultiSessionPersistenceMetric":
        """
        Load configuration from a session snapshot saved by
        SingleSessionPoisoningMetric.

        The snapshot contains: experiment_id, poison_rate, poison_indices.
        Use this to continue from where Consolidation Eval Suite Test 4 left off.
        """
        with open(snapshot_path) as f:
            snapshot = json.load(f)
        logger.info(
            f"[persistence] Loading from snapshot: {snapshot_path}\n"
            f"  experiment_id: {snapshot.get('experiment_id')}\n"
            f"  poison_rate: {snapshot.get('poison_rate')}\n"
            f"  n_poisoned_docs: {len(snapshot.get('poison_indices', []))}"
        )
        return cls(**kwargs)

    def _compute_persistence_rate(
        self,
        retrieval_results: list[RetrievalResult],
        target_doc_ids: set[str],
    ) -> float:
        """
        Fraction of queries where at least one target doc appears in top-K.
        """
        if not target_doc_ids or not retrieval_results:
            return 0.0
        hits = sum(
            1 for r in retrieval_results
            if set(r.retrieved_ids[:self.k]).intersection(target_doc_ids)
        )
        return hits / len(retrieval_results)

    def _run_ragas_sample(
        self,
        cases: list[RetrievalCase],
        retrieval_results: list[RetrievalResult],
        sample_n: int = 10,
    ) -> float | None:
        """
        Run RAGAS faithfulness on a sample of cases for efficiency.
        Full dataset per cycle is expensive — sample_n cases sufficient.
        """
        try:
            from eval_engine.metrics.ragas_adapter import RagasCase
            from eval_engine.metrics.ragas_runner import run_ragas_evaluation

            result_map = {r.query_id: r for r in retrieval_results}
            sample = cases[:sample_n]
            ragas_cases = []

            for case in sample:
                r = result_map.get(case.query_id)
                if r:
                    ragas_cases.append(RagasCase(
                        question=case.query,
                        answer=" ".join(r.retrieved_ids[:3]),
                        contexts=r.retrieved_ids[:5],
                    ))

            if not ragas_cases:
                return None

            scores = run_ragas_evaluation(ragas_cases, return_partial=True)
            return float(scores.get("faithfulness", 0.0))

        except Exception as e:
            logger.warning(f"[persistence] RAGAS faithfulness failed: {e}")
            return None
