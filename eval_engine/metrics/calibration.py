"""
eval_engine/metrics/calibration.py

Confidence Calibration Evaluation

PURPOSE:
    Measures whether stated confidence scores match actual correctness rates.
    Answers: "When the proposer says 90% confidence, is it right 90%
    of the time?"

    This is technically distinct from faithfulness and grounding:
        - Faithfulness: is the answer supported by context?
        - Calibration:  does the confidence score accurately predict
                        whether the answer is correct?

    A system can be highly faithful but poorly calibrated (always says 0.95
    confidence regardless of actual accuracy). Calibration exposes this.

METRICS IMPLEMENTED:
    1. Expected Calibration Error (ECE)
       Industry standard calibration metric.
       Bins predictions by confidence, measures gap between
       confidence and actual accuracy per bin.
       ECE = 0.0 → perfect calibration
       ECE = 1.0 → maximally miscalibrated

    2. Brier Score
       Mean squared error between confidence and binary correctness.
       Brier = 0.0 → perfect
       Brier = 1.0 → worst possible
       Proper scoring rule — cannot be gamed by confidence hedging.

    3. Reliability Diagram Data
       Bin-level (confidence, accuracy) pairs for plotting.
       Used in paper figures and appendix.

    4. Overconfidence / Underconfidence classification
       Overconfident: mean confidence > mean accuracy
       Underconfident: mean confidence < mean accuracy

INTEGRATION:
    Feeds into TrustScore composite (trust_score.py).
    Most naturally attached to ProposerAgent output in Phase 3 pipeline.

    Usage:
        from eval_engine.metrics.calibration import CalibrationMetric

        metric = CalibrationMetric(n_bins=10)
        result = metric.score_from_pairs(
            confidence_scores=[0.9, 0.7, 0.8, 0.6],
            correctness_labels=[1, 1, 0, 0],   # 1=correct, 0=incorrect
        )
        print(result.metadata["ece"])
        print(result.metadata["brier_score"])

PAPER NOTE:
    ECE requires enough samples per bin to be meaningful.
    Minimum recommended: 50 samples with n_bins=10 (5 per bin).
    Flag results with <5 samples per bin as unreliable in paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBin:
    """Single confidence bin for reliability diagram."""
    bin_lower: float
    bin_upper: float
    mean_confidence: float
    mean_accuracy: float
    n_samples: int
    calibration_gap: float      # mean_confidence - mean_accuracy
    reliable: bool              # True if n_samples >= min_samples_per_bin


@dataclass
class CalibrationReport:
    """Full calibration analysis output."""
    ece: float
    brier_score: float
    mean_confidence: float
    mean_accuracy: float
    overconfident: bool           # mean_confidence > mean_accuracy
    bins: list[CalibrationBin]
    n_samples: int
    n_bins: int
    reliable: bool                # True if enough samples per bin overall

    @property
    def calibration_direction(self) -> str:
        if abs(self.mean_confidence - self.mean_accuracy) < 0.02:
            return "well_calibrated"
        return "overconfident" if self.overconfident else "underconfident"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": round(self.ece, 6),
            "brier_score": round(self.brier_score, 6),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "calibration_direction": self.calibration_direction,
            "overconfident": self.overconfident,
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "reliable": self.reliable,
            "bins": [
                {
                    "range": f"[{b.bin_lower:.1f}, {b.bin_upper:.1f})",
                    "mean_confidence": round(b.mean_confidence, 4),
                    "mean_accuracy": round(b.mean_accuracy, 4),
                    "gap": round(b.calibration_gap, 4),
                    "n": b.n_samples,
                    "reliable": b.reliable,
                }
                for b in self.bins
            ],
        }

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print("  CALIBRATION REPORT")
        print(f"{'='*55}")
        print(f"  ECE:               {self.ece:.4f}  (0=perfect, 1=worst)")
        print(f"  Brier Score:       {self.brier_score:.4f}  (0=perfect, 1=worst)")
        print(f"  Mean Confidence:   {self.mean_confidence:.4f}")
        print(f"  Mean Accuracy:     {self.mean_accuracy:.4f}")
        print(f"  Direction:         {self.calibration_direction}")
        print(f"  N samples:         {self.n_samples}")
        print(f"  Reliable:          {self.reliable}")
        print("\n  Reliability Diagram:")
        print(f"  {'Bin':<18} {'Conf':>6} {'Acc':>6} {'Gap':>7} {'N':>5}")
        print(f"  {'-'*45}")
        for b in self.bins:
            flag = " *" if not b.reliable else ""
            print(
                f"  [{b.bin_lower:.1f}, {b.bin_upper:.1f}){flag:<4}"
                f"  {b.mean_confidence:>6.3f}"
                f"  {b.mean_accuracy:>6.3f}"
                f"  {b.calibration_gap:>+7.3f}"
                f"  {b.n_samples:>5}"
            )
        if any(not b.reliable for b in self.bins):
            print("  * fewer than 5 samples — bin unreliable")
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def compute_ece(
    confidence_scores: list[float],
    correctness_labels: list[int],
    n_bins: int = 10,
    min_samples_per_bin: int = 5,
) -> tuple[float, list[CalibrationBin]]:
    """
    Compute Expected Calibration Error.

    Args:
        confidence_scores:   Predicted confidence per sample (0.0–1.0)
        correctness_labels:  Binary correctness per sample (1=correct, 0=wrong)
        n_bins:              Number of equal-width confidence bins
        min_samples_per_bin: Bins with fewer samples flagged as unreliable

    Returns:
        (ece, bins)
    """
    confs = np.array(confidence_scores, dtype=float)
    labels = np.array(correctness_labels, dtype=float)
    n = len(confs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        # Include upper bound only in last bin
        if i < n_bins - 1:
            mask = (confs >= lower) & (confs < upper)
        else:
            mask = (confs >= lower) & (confs <= upper)

        n_bin = int(mask.sum())
        if n_bin == 0:
            bins.append(CalibrationBin(
                bin_lower=lower, bin_upper=upper,
                mean_confidence=0.0, mean_accuracy=0.0,
                n_samples=0, calibration_gap=0.0,
                reliable=False,
            ))
            continue

        mean_conf = float(confs[mask].mean())
        mean_acc = float(labels[mask].mean())
        gap = mean_conf - mean_acc

        ece += (n_bin / n) * abs(gap)

        bins.append(CalibrationBin(
            bin_lower=lower, bin_upper=upper,
            mean_confidence=mean_conf,
            mean_accuracy=mean_acc,
            n_samples=n_bin,
            calibration_gap=gap,
            reliable=n_bin >= min_samples_per_bin,
        ))

    return float(ece), bins


def compute_brier_score(
    confidence_scores: list[float],
    correctness_labels: list[int],
) -> float:
    """
    Compute Brier Score = mean((confidence - correctness)^2).
    Proper scoring rule — lower is better.
    """
    confs = np.array(confidence_scores, dtype=float)
    labels = np.array(correctness_labels, dtype=float)
    return float(np.mean((confs - labels) ** 2))


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class CalibrationMetric(BaseMetric):
    """
    Confidence calibration evaluation — ECE + Brier Score.

    Two usage modes:

    1. Single-query mode (score()):
       Scores one query using binary ground truth match as correctness.
       Accumulates (confidence, correctness) pairs across calls.
       Call compute_calibration_report() after all queries for ECE/Brier.

    2. Batch mode (score_from_pairs()):
       Pass all confidence scores and correctness labels at once.
       Returns CalibrationReport directly.

    Mode 2 is recommended for batch evaluation — run all debate rounds first,
    collect (proposer_confidence, judge_verdict) pairs, then compute ECE.
    """

    name = "calibration"

    def __init__(
        self,
        n_bins: int = 10,
        min_samples_per_bin: int = 5,
        correctness_threshold: float = 0.5,
    ) -> None:
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.correctness_threshold = correctness_threshold
        # Accumulates across single-query calls
        self._confidence_buffer: list[float] = []
        self._correctness_buffer: list[int] = []

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        confidence: float | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Single-query calibration data point.

        Correctness inferred from ground_truth match if confidence provided.
        Accumulates into internal buffer — call compute_calibration_report()
        after all queries.

        Args:
            confidence: Proposer's stated confidence (0.0–1.0).
                        If None, returns error result.
            ground_truth: Used to determine correctness (string match proxy).
        """
        self.validate_inputs(question, contexts, answer)

        if confidence is None:
            return self.error_result(
                "calibration requires confidence kwarg — "
                "pass proposer.parsed['confidence'] here"
            )

        if not 0.0 <= confidence <= 1.0:
            return self.error_result(
                f"confidence must be in [0,1], got {confidence}"
            )

        # Determine correctness
        if ground_truth:
            correct = int(ground_truth.lower() in answer.lower())
        else:
            # Without ground truth, use confidence > threshold as proxy
            # Flag this as unreliable in metadata
            correct = int(confidence >= self.correctness_threshold)
            logger.warning(
                "[calibration] No ground_truth — using confidence threshold "
                "as correctness proxy. Results unreliable without ground truth."
            )

        self._confidence_buffer.append(confidence)
        self._correctness_buffer.append(correct)

        return MetricResult(
            metric_name=self.name,
            score=abs(confidence - correct),   # Per-sample calibration error
            raw={
                "confidence": confidence,
                "correctness": correct,
                "calibration_error": abs(confidence - correct),
            },
            metadata={
                "has_ground_truth": ground_truth is not None,
                "buffer_size": len(self._confidence_buffer),
                "note": "Call compute_calibration_report() for ECE/Brier after all queries.",
            },
        )

    def score_from_pairs(
        self,
        confidence_scores: list[float],
        correctness_labels: list[int],
        label: str = "calibration",
    ) -> MetricResult:
        """
        Batch calibration — pass all pairs at once.

        Args:
            confidence_scores:  List of confidence values (0.0–1.0)
            correctness_labels: List of binary correctness (1=correct, 0=wrong)
            label:              Identifier for this calibration run

        Returns:
            MetricResult with score=ECE and full CalibrationReport in raw.
        """
        if len(confidence_scores) != len(correctness_labels):
            return self.error_result(
                f"confidence_scores and correctness_labels must be same length: "
                f"{len(confidence_scores)} vs {len(correctness_labels)}"
            )
        if len(confidence_scores) < self.n_bins:
            logger.warning(
                f"[calibration] n_samples={len(confidence_scores)} < n_bins={self.n_bins}. "
                f"ECE will be unreliable. Recommend at least {self.n_bins * self.min_samples_per_bin} samples."
            )

        ece, bins = compute_ece(
            confidence_scores, correctness_labels,
            self.n_bins, self.min_samples_per_bin,
        )
        brier = compute_brier_score(confidence_scores, correctness_labels)

        mean_conf = float(np.mean(confidence_scores))
        mean_acc = float(np.mean(correctness_labels))
        n_reliable = sum(1 for b in bins if b.reliable)
        reliable = n_reliable >= (self.n_bins * 0.7)  # 70% of bins must be reliable

        report = CalibrationReport(
            ece=ece,
            brier_score=brier,
            mean_confidence=mean_conf,
            mean_accuracy=mean_acc,
            overconfident=mean_conf > mean_acc,
            bins=bins,
            n_samples=len(confidence_scores),
            n_bins=self.n_bins,
            reliable=reliable,
        )
        report.print_summary()

        return MetricResult(
            metric_name=self.name,
            score=ece,   # Primary scalar: ECE (lower = better)
            raw=report.to_dict(),
            metadata={
                "label": label,
                "reliable": reliable,
                "n_samples": len(confidence_scores),
                "brier_score": brier,
                "calibration_direction": report.calibration_direction,
            },
        )

    def compute_calibration_report(self) -> MetricResult:
        """
        Compute ECE/Brier from accumulated single-query calls.
        Call this after processing all queries with score().
        """
        if not self._confidence_buffer:
            return self.error_result(
                "No data accumulated. Call score() for each query first, "
                "or use score_from_pairs() for batch mode."
            )
        result = self.score_from_pairs(
            self._confidence_buffer,
            self._correctness_buffer,
            label="accumulated",
        )
        self._confidence_buffer.clear()
        self._correctness_buffer.clear()
        return result

    def reset(self) -> None:
        """Clear accumulated buffer."""
        self._confidence_buffer.clear()
        self._correctness_buffer.clear()
