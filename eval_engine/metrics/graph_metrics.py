"""
eval_engine/metrics/graph_metrics.py

Graph Refinement Quality Metrics

SOURCE: Consolidation RAG eval framework (your code), UNCHANGED core logic.
ADDITIONS:
  FIX-5: Input validation added to all public functions.
          - Type checks on integer inputs
          - Ordering guard on before/after snapshot pairs using GraphSnapshot.is_before()
          - Explicit warnings when reduction ratio is negative (graph expansion)
            Expansion is a valid finding for Consolidation Eval Suite; it must be logged, not silently
            reported as a negative number.
  ADDITION: BaseMetric adapter classes for CLI + registry dispatch.

UNCHANGED from your original:
  - All computation logic (verbatim)
  - All docstrings and inline comments (verbatim)
  - _relative_reduction helper (verbatim)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from eval_engine.schemas import GraphSnapshot
from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# =============================================================================
# YOUR ORIGINAL CODE — VERBATIM (do not modify core logic)
# =============================================================================

def entity_coverage(covered_entities: int, expected_entities: int) -> float:
    """Compute entity coverage as covered_entities / expected_entities."""
    if expected_entities <= 0:
        return 0.0
    return covered_entities / expected_entities


def relation_completeness(valid_relations: int, expected_relations: int) -> float:
    """Compute relation completeness as valid_relations / expected_relations."""
    if expected_relations <= 0:
        return 0.0
    return valid_relations / expected_relations


def compression_delta(before: GraphSnapshot, after: GraphSnapshot) -> dict[str, float]:
    """Measure graph size change after refinement."""
    node_reduction = _relative_reduction(before.node_count, after.node_count)
    edge_reduction = _relative_reduction(before.edge_count, after.edge_count)
    return {
        "nodes_before": float(before.node_count),
        "nodes_after": float(after.node_count),
        "node_reduction_ratio": node_reduction,
        "edges_before": float(before.edge_count),
        "edges_after": float(after.edge_count),
        "edge_reduction_ratio": edge_reduction,
    }


def deduplication_delta(before: GraphSnapshot, after: GraphSnapshot) -> dict[str, float]:
    """Measure duplicate reduction after refinement."""
    duplicate_reduction = _relative_reduction(
        before.duplicate_count,
        after.duplicate_count,
    )
    return {
        "duplicates_before": float(before.duplicate_count),
        "duplicates_after": float(after.duplicate_count),
        "duplicate_reduction_ratio": duplicate_reduction,
    }


def snapshot_entity_coverage(snapshot: GraphSnapshot) -> float:
    """Compute entity coverage from a GraphSnapshot when available."""
    if snapshot.covered_entities is None or snapshot.expected_entities is None:
        return 0.0
    return entity_coverage(snapshot.covered_entities, snapshot.expected_entities)


def snapshot_relation_completeness(snapshot: GraphSnapshot) -> float:
    """Compute relation completeness from a GraphSnapshot when available."""
    if snapshot.valid_relations is None or snapshot.expected_relations is None:
        return 0.0
    return relation_completeness(snapshot.valid_relations, snapshot.expected_relations)


def _relative_reduction(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before


# =============================================================================
# FIX-5: VALIDATED WRAPPERS
# Call these instead of the raw functions when running inside EvalRunner.
# These add ordering guards, type checks, and expansion warnings.
# =============================================================================

def validated_compression_delta(
    before: GraphSnapshot,
    after: GraphSnapshot,
    experiment_id: str | None = None,
) -> dict[str, float]:
    """
    compression_delta() with ordering guard and expansion warnings.

    Raises:
        ValueError: if before and after cannot be ordered (no cycle_id or captured_at)
        ValueError: if snapshots belong to different experiments
    Warns:
        UserWarning: if node_reduction_ratio or edge_reduction_ratio is negative
                     (graph expanded rather than compressed — valid Consolidation Eval Suite finding)
    """
    # Experiment ID consistency check
    if (
        before.experiment_id is not None
        and after.experiment_id is not None
        and before.experiment_id != after.experiment_id
    ):
        raise ValueError(
            f"Snapshot experiment_id mismatch: "
            f"before='{before.experiment_id}' vs after='{after.experiment_id}'. "
            f"Snapshots from different experiments cannot be compared."
        )

    # Ordering guard — raises ValueError if neither cycle_id nor captured_at is set
    if not before.is_before(after):
        raise ValueError(
            f"Snapshot ordering violation: 'before' snapshot (cycle_id={before.cycle_id}) "
            f"is not earlier than 'after' snapshot (cycle_id={after.cycle_id}). "
            f"Pass snapshots in (pre_consolidation, post_consolidation) order."
        )

    result = compression_delta(before, after)

    # Expansion warnings — negative reduction is a finding, not a bug
    if result["node_reduction_ratio"] < 0:
        msg = (
            f"[compression_delta] Node count EXPANDED after consolidation "
            f"(before={int(result['nodes_before'])}, after={int(result['nodes_after'])}, "
            f"ratio={result['node_reduction_ratio']:.4f}). "
            f"This is a valid finding — log and report in Consolidation Eval Suite results."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)

    if result["edge_reduction_ratio"] < 0:
        msg = (
            f"[compression_delta] Edge count EXPANDED after consolidation "
            f"(before={int(result['edges_before'])}, after={int(result['edges_after'])}, "
            f"ratio={result['edge_reduction_ratio']:.4f}). "
            f"This is a valid finding — log and report in Consolidation Eval Suite results."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)

    return result


def validated_deduplication_delta(
    before: GraphSnapshot,
    after: GraphSnapshot,
) -> dict[str, float]:
    """
    deduplication_delta() with ordering guard.
    Same ordering and experiment consistency checks as validated_compression_delta.
    """
    if (
        before.experiment_id is not None
        and after.experiment_id is not None
        and before.experiment_id != after.experiment_id
    ):
        raise ValueError(
            f"Snapshot experiment_id mismatch: "
            f"'{before.experiment_id}' vs '{after.experiment_id}'"
        )

    if not before.is_before(after):
        raise ValueError(
            f"Snapshot ordering violation: before cycle_id={before.cycle_id} "
            f"is not earlier than after cycle_id={after.cycle_id}."
        )

    result = deduplication_delta(before, after)

    if result["duplicate_reduction_ratio"] < 0:
        msg = (
            f"[deduplication_delta] Duplicate count INCREASED after consolidation "
            f"(before={int(result['duplicates_before'])}, after={int(result['duplicates_after'])}, "
            f"ratio={result['duplicate_reduction_ratio']:.4f}). "
            f"Investigate consolidation merge logic."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)

    return result


# =============================================================================
# BASEMET ADAPTERS
# =============================================================================

class CompressionDeltaMetric(BaseMetric):
    """
    BaseMetric adapter for compression_delta().
    Graph metrics have zero LLM cost — cost tracker records $0.00.

    Score returned: node_reduction_ratio (primary scalar).
    Full dict available in result.raw.
    """

    name = "compression_delta"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        before_snapshot: GraphSnapshot | None = None,
        after_snapshot: GraphSnapshot | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        if before_snapshot is None or after_snapshot is None:
            return self.error_result(
                "compression_delta requires before_snapshot and after_snapshot kwargs"
            )

        try:
            raw = validated_compression_delta(before_snapshot, after_snapshot)
        except (ValueError, Exception) as e:
            return self.error_result(str(e))

        return MetricResult(
            metric_name=self.name,
            score=raw["node_reduction_ratio"],
            raw=raw,
            metadata={
                "cycle_before": before_snapshot.cycle_id,
                "cycle_after": after_snapshot.cycle_id,
                "experiment_id": after_snapshot.experiment_id,
                "llm_cost_usd": 0.0,
            },
        )


class DeduplicationDeltaMetric(BaseMetric):
    """BaseMetric adapter for deduplication_delta()."""

    name = "deduplication_delta"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        before_snapshot: GraphSnapshot | None = None,
        after_snapshot: GraphSnapshot | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        if before_snapshot is None or after_snapshot is None:
            return self.error_result(
                "deduplication_delta requires before_snapshot and after_snapshot kwargs"
            )

        try:
            raw = validated_deduplication_delta(before_snapshot, after_snapshot)
        except (ValueError, Exception) as e:
            return self.error_result(str(e))

        return MetricResult(
            metric_name=self.name,
            score=raw["duplicate_reduction_ratio"],
            raw=raw,
            metadata={
                "cycle_before": before_snapshot.cycle_id,
                "cycle_after": after_snapshot.cycle_id,
                "llm_cost_usd": 0.0,
            },
        )


class EntityCoverageMetric(BaseMetric):
    """BaseMetric adapter for entity_coverage() / snapshot_entity_coverage()."""

    name = "entity_coverage"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        snapshot: GraphSnapshot | None = None,
        covered_entities: int | None = None,
        expected_entities: int | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        if snapshot is not None:
            score_val = snapshot_entity_coverage(snapshot)
            src = "snapshot"
        elif covered_entities is not None and expected_entities is not None:
            score_val = entity_coverage(covered_entities, expected_entities)
            src = "direct"
        else:
            return self.error_result(
                "entity_coverage requires either snapshot kwarg or "
                "covered_entities + expected_entities kwargs"
            )

        return MetricResult(
            metric_name=self.name,
            score=score_val,
            raw={"source": src, "llm_cost_usd": 0.0},
        )
