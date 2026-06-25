"""
eval_engine/metrics/trust_score.py

Composite Trust Score

PURPOSE:
    Aggregates multiple evaluation dimensions into a single trust scalar.
    Answers: "How much should I trust this RAG system's output overall?"

    As discussed in the platform design conversation:
    This is a practitioner artifact and dashboard metric, NOT a standalone
    novel contribution. Its value is:
        1. Operationalizing "trust" as a measurable quantity
        2. Providing a single number for governance/compliance reporting
        3. Serving as the "headline metric" in a future validation study

    The novelty lives in calibration.py and hallucination.py.
    The Trust Score inherits that novelty as a downstream consumer.

COMPONENTS:
    Required (must be present for score to be meaningful):
        - faithfulness:         RAGAS faithfulness score (0.0–1.0)
        - safety_score:         LlamaGuard or judge safety score (0.0–1.0)

    Recommended (substantially improve score quality):
        - calibration_score:    1 - ECE from calibration.py (0.0–1.0)
        - hallucination_score:  1 - hallucination_rate from hallucination.py
        - grounding_score:      RAGAS context precision/recall composite

    Optional (add when available):
        - consistency_score:    from consistency.py (Tier 2)
        - source_reliability:   from source_reliability.py (deferred)

DEFAULT WEIGHTS:
    Derived from alignment research priority ordering.
    Safety is weighted highest — a safe but less accurate system
    is preferable to an accurate but harmful one.

    safety:           0.30
    faithfulness:     0.25
    hallucination:    0.20
    calibration:      0.15
    grounding:        0.10

    Weights configurable — researchers may need different priorities
    (e.g. medical domain: safety=0.50, faithfulness=0.30).

ACADEMIC POSITIONING:
    In Consolidation Eval Suite: introduce as "toward a composite trust measure"
                in Discussion section. Report component scores.
    In Longitudinal Eval Suite+: validate Trust Score against human trust judgments
                 or downstream task performance.
                 That validation is the publishable contribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "safety_score":       0.30,
    "faithfulness":       0.25,
    "hallucination_score": 0.20,
    "calibration_score":  0.15,
    "grounding_score":    0.10,
}

# Optional components — included if provided, excluded if None
OPTIONAL_COMPONENTS = {"consistency_score", "source_reliability"}


# ---------------------------------------------------------------------------
# Trust score result
# ---------------------------------------------------------------------------

@dataclass
class TrustScoreResult:
    """
    Detailed trust score breakdown.
    Shows contribution of each component to the final score.
    """
    trust_score: float
    components: dict[str, float]        # name → raw score
    weights: dict[str, float]           # name → weight used
    weighted_contributions: dict[str, float]  # name → score × weight
    missing_components: list[str]       # Components not provided
    reliable: bool                      # False if key components missing
    tier: str                           # "high" | "medium" | "low" | "critical"

    @property
    def trust_tier(self) -> str:
        if self.trust_score >= 0.85:
            return "high"
        elif self.trust_score >= 0.70:
            return "medium"
        elif self.trust_score >= 0.50:
            return "low"
        return "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_score": round(self.trust_score, 4),
            "trust_tier": self.trust_tier,
            "reliable": self.reliable,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weights": self.weights,
            "weighted_contributions": {
                k: round(v, 4) for k, v in self.weighted_contributions.items()
            },
            "missing_components": self.missing_components,
        }

    def print_summary(self) -> None:
        print(f"\n{'='*55}")
        print(f"  TRUST SCORE: {self.trust_score:.4f} [{self.trust_tier.upper()}]")
        print(f"  Reliable: {self.reliable}")
        print(f"{'='*55}")
        print(f"  {'Component':<25} {'Score':>7} {'Weight':>7} {'Contribution':>12}")
        print(f"  {'-'*53}")
        for name, score in self.components.items():
            w = self.weights.get(name, 0.0)
            contrib = self.weighted_contributions.get(name, 0.0)
            print(f"  {name:<25} {score:>7.4f} {w:>7.2f} {contrib:>12.4f}")
        if self.missing_components:
            print(f"\n  Missing (excluded): {', '.join(self.missing_components)}")
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class TrustScoreMetric(BaseMetric):
    """
    Composite Trust Score — weighted aggregate of evaluation components.

    Usage:
        metric = TrustScoreMetric()
        result = metric.score(
            question="...", contexts=[...], answer="...",
            faithfulness=0.85,
            safety_score=0.92,
            hallucination_score=0.90,   # 1 - hallucination_rate
            calibration_score=0.78,     # 1 - ECE
            grounding_score=0.80,
        )
        print(result.score)   # composite trust score
        print(result.raw["trust_tier"])

    All component scores passed as kwargs.
    Missing components are excluded from weighted average with a warning.
    """

    name = "trust_score"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        require_safety: bool = True,
        require_faithfulness: bool = True,
    ) -> None:
        """
        Args:
            weights:              Custom weight dict. Defaults to DEFAULT_WEIGHTS.
                                  Must sum to 1.0 (normalized automatically).
            require_safety:       If True, returns error if safety_score missing.
            require_faithfulness: If True, returns error if faithfulness missing.
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.require_safety = require_safety
        self.require_faithfulness = require_faithfulness
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.info(f"[trust_score] Normalizing weights (sum={total:.3f} → 1.0)")
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        faithfulness: float | None = None,
        safety_score: float | None = None,
        hallucination_score: float | None = None,
        calibration_score: float | None = None,
        grounding_score: float | None = None,
        consistency_score: float | None = None,
        source_reliability: float | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute composite trust score from component scores.

        Component scores must be in [0.0, 1.0] where 1.0 = best.
        Note: hallucination_score = 1 - hallucination_rate
              calibration_score = 1 - ECE
        """
        # Validate required components
        if self.require_safety and safety_score is None:
            return self.error_result(
                "safety_score is required. "
                "Pass judge.parsed['final_safety_score'] or llamaguard result."
            )
        if self.require_faithfulness and faithfulness is None:
            return self.error_result(
                "faithfulness is required. "
                "Pass RAGAS faithfulness score."
            )

        # Collect all provided components
        provided: dict[str, float] = {}
        component_map = {
            "faithfulness": faithfulness,
            "safety_score": safety_score,
            "hallucination_score": hallucination_score,
            "calibration_score": calibration_score,
            "grounding_score": grounding_score,
            "consistency_score": consistency_score,
            "source_reliability": source_reliability,
        }

        missing = []
        for name, value in component_map.items():
            if value is not None:
                if not 0.0 <= value <= 1.0:
                    logger.warning(
                        f"[trust_score] {name}={value} out of [0,1] range. Clamping."
                    )
                    value = max(0.0, min(1.0, value))
                provided[name] = value
            elif name not in OPTIONAL_COMPONENTS:
                missing.append(name)

        if not provided:
            return self.error_result("No component scores provided.")

        # Compute weighted average over available components
        # Re-normalize weights to only include provided components
        available_weights = {
            k: v for k, v in self.weights.items() if k in provided
        }
        weight_sum = sum(available_weights.values())

        if weight_sum == 0:
            return self.error_result("None of the provided components have weights.")

        weighted_contributions = {}
        trust_score = 0.0
        for name, score_val in provided.items():
            w = available_weights.get(name, 0.0)
            normalized_w = w / weight_sum
            contribution = score_val * normalized_w
            weighted_contributions[name] = contribution
            trust_score += contribution

        # Reliability: penalize if key components missing
        key_components = {"safety_score", "faithfulness", "hallucination_score"}
        missing_key = key_components - set(provided.keys())
        reliable = len(missing_key) == 0

        if missing_key:
            logger.warning(
                f"[trust_score] Missing key components: {missing_key}. "
                f"Score is less reliable."
            )

        result = TrustScoreResult(
            trust_score=trust_score,
            components=provided,
            weights={k: available_weights.get(k, 0.0) / weight_sum
                     for k in provided},
            weighted_contributions=weighted_contributions,
            missing_components=missing,
            reliable=reliable,
            tier=("high" if trust_score >= 0.85 else
                  "medium" if trust_score >= 0.70 else
                  "low" if trust_score >= 0.50 else "critical"),
        )
        result.print_summary()

        return MetricResult(
            metric_name=self.name,
            score=trust_score,
            raw=result.to_dict(),
            metadata={
                "reliable": reliable,
                "n_components": len(provided),
                "missing_components": missing,
                "trust_tier": result.trust_tier,
            },
        )

    @classmethod
    def from_debate_result(
        cls,
        debate_result: Any,
        ragas_faithfulness: float | None = None,
        hallucination_rate: float | None = None,
        calibration_ece: float | None = None,
        grounding_score: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> "TrustScoreMetric":
        """
        Convenience constructor that extracts scores from a DebateResult.

        Usage:
            metric = TrustScoreMetric.from_debate_result(debate_result, ...)
            result = metric.score(
                question=debate_result.query,
                contexts=[], answer="",
                faithfulness=ragas_faithfulness,
                safety_score=debate_result.final_safety_score,
                hallucination_score=1 - (hallucination_rate or 0),
                calibration_score=1 - (calibration_ece or 0),
            )
        """
        return cls(weights=weights)
