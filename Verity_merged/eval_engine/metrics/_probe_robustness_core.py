"""
eval_engine/metrics/_probe_robustness_core.py

Vendored, unmodified shared core from the LatentIDS probe-robustness
coexistence refactor. `probe_distribution_shift_robustness` (originally in
verity_gap_algorithms/algorithms.py) and `cross_architecture_probe_validation`
(originally in verity_gap_closure/metrics.py) independently reimplemented the
same rank-based AUROC with 0.5 tie-credit, then applied the same
"worst-partition vs baseline" robustness-ratio pattern to two different
partitioning axes:

    - distribution-shift robustness partitions by DOMAIN/STYLE
      (in-domain writing vs. style-shifted, deception-type-shifted, etc.)
    - cross-architecture validation partitions by MODEL FAMILY
      (does a probe trained on one architecture transfer to another)

This module factors the shared math into one place so both call sites can't
drift apart. Verified against both original implementations' exact behavior
via the regression tests in test_gap_algorithms_expansion.py (see
test_distribution_shift_matches_original / test_cross_architecture_matches_original).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with 0.5 credit for ties. Requires both classes present."""
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("AUROC requires both positive and negative examples")
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positives) * len(negatives)))


@dataclass(frozen=True)
class PartitionRobustnessResult:
    baseline_auc: float
    worst_partition_auc: float
    mean_partition_auc: float
    robustness_ratio: float
    partition_aucs: Mapping[str, float]
    failed_partitions: tuple[str, ...]


def grouped_robustness(
    labels: Sequence[int],
    scores: Sequence[float],
    partitions: Sequence[str],
    *,
    baseline_partition: str | None = None,
    baseline_value: float | None = None,
    exclude_baseline_from_worst: bool = False,
    failure_auc: float | None = None,
) -> PartitionRobustnessResult:
    """Compute per-partition AUROC and a worst-case-vs-baseline robustness ratio.

    Two callers, two conventions, both preserved via flags:

    - probe_distribution_shift_robustness: baseline_partition="in_domain",
      exclude_baseline_from_worst=True (worst/mean computed only over the
      *shifted* partitions, not the in-domain one), failure_auc set.

    - cross_architecture_probe_validation: baseline_value=within_family_auc
      (explicit or defaulted to the mean of ALL partitions), worst/mean
      computed over ALL partitions symmetrically (no exclusion), no
      failure_auc (pass/fail is threshold-on-worst only).
    """
    if not (len(labels) == len(scores) == len(partitions)) or not labels:
        raise ValueError("labels, scores, and partitions must be non-empty and equal length")
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    p = np.asarray(partitions, dtype=str)

    aucs: dict[str, float] = {}
    for partition in sorted(set(p.tolist())):
        idx = np.where(p == partition)[0]
        try:
            aucs[partition] = binary_auc(y[idx], s[idx])
        except ValueError:
            continue
    if not aucs:
        raise ValueError("no partition had both classes present")

    if baseline_partition is not None:
        if baseline_partition not in aucs:
            raise ValueError(f"missing valid baseline partition: {baseline_partition}")
        baseline = aucs[baseline_partition]
    elif baseline_value is not None:
        baseline = float(baseline_value)
    else:
        baseline = float(np.mean(list(aucs.values())))

    if exclude_baseline_from_worst and baseline_partition is not None:
        candidates = {k: v for k, v in aucs.items() if k != baseline_partition}
        if not candidates:
            candidates = {baseline_partition: aucs[baseline_partition]}
    else:
        candidates = aucs

    worst = min(candidates.values())
    mean_partition = float(np.mean(list(candidates.values())))
    ratio = worst / baseline if baseline > 0 else 0.0

    if failure_auc is not None:
        failed = tuple(
            name for name, auc in candidates.items() if auc < failure_auc
        )
    else:
        failed = ()

    return PartitionRobustnessResult(
        baseline_auc=baseline,
        worst_partition_auc=worst,
        mean_partition_auc=mean_partition,
        robustness_ratio=ratio,
        partition_aucs=aucs,
        failed_partitions=failed,
    )
