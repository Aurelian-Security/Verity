"""
eval_engine/metrics/goal_misgeneralization.py

Goal Misgeneralization Detection — SCAFFOLD

STATUS: Scaffold. Interface locked. Implementation deferred to Tier 3.

RESEARCH QUESTION:
    Did Consolidation RAG optimize a proxy objective instead of the intended
    objective during consolidation?

    Example:
        Intended:  Retrieve the most accurate information
        Proxy:     Retrieve the most frequently co-occurring information
        Failure:   After consolidation, popular-but-wrong answers
                   surface more readily than rare-but-correct ones

    This is distinct from reward hacking (Phase 3):
        - Reward hacking: model games the evaluation metric at inference
        - Goal misgeneralization: model learned wrong objective during training/
                                  consolidation — behaves correctly in-distribution
                                  but fails on distribution shift

THEORETICAL FRAMEWORK:
    Mesa-optimization (Hubinger et al., 2019) — "Risks from Learned Optimization"
    The consolidation cycle is a form of in-context learning / self-modification.
    Goal misgeneralization asks whether the consolidation objective
    (improve retrieval quality) transferred correctly.

PLANNED DETECTION APPROACH:
    1. In-distribution test: standard eval queries → measure accuracy
    2. Out-of-distribution test: adversarially shifted queries → measure accuracy
    3. Misgeneralization indicator: large accuracy gap between (1) and (2)
       when the model "should" generalize but doesn't

    Specific to Consolidation RAG:
    - Test whether consolidation-improved retrieval generalizes to
      query phrasings not seen during consolidation
    - Measure whether graph pruning removed semantically important
      but statistically rare knowledge

IMPLEMENTATION REQUIREMENTS:
    - Distribution shift test set (in-distribution + OOD query pairs)
    - Consolidation RAG runtime integration for pre/post consolidation comparison
    - Statistical framework for significance of accuracy gap

PAPER PLACEMENT:
    Advanced Eval Suite or independent paper — "Detecting Goal Misgeneralization in
    Self-Consolidating RAG Systems"
    Target: NeurIPS Safety Workshop, ICML Alignment Forum, ARC/METR venues.

PRIOR WORK TO CITE:
    - Hubinger et al. (2019) — Risks from Learned Optimization (mesa-optimization)
    - Langosco et al. (2022) — Goal Misgeneralization in Deep RL
    - Shah et al. (2022) — Goal Misgeneralization: Why Correct Specifications
      Aren't Enough for Correct Goals
"""

from __future__ import annotations

import logging
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class GoalMisgeneralizationMetric(BaseMetric):
    """
    Goal Misgeneralization Detection.

    SCAFFOLD — returns NotImplemented result with research context.
    Implementation target: Tier 3.

    This is frontier alignment research. Do not implement until:
    1. Consolidation RAG Evaluation + 2 are published
    2. A distribution shift test set is designed and validated
    3. The mesa-optimization framing is reviewed by the research team
    """

    name = "goal_misgeneralization"

    def __init__(
        self,
        ood_shift_type: str = "query_paraphrase",
        accuracy_gap_threshold: float = 0.15,
    ) -> None:
        self.ood_shift_type = ood_shift_type
        self.accuracy_gap_threshold = accuracy_gap_threshold

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        SCAFFOLD — not yet implemented.

        When implemented, compares in-distribution vs OOD accuracy
        to detect objective misgeneralization during consolidation.
        """
        logger.info(
            "[goal_misgeneralization] SCAFFOLD — implementation deferred to Tier 3. "
            "Requires: Consolidation RAG runtime, OOD test set, mesa-optimization framework."
        )
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            raw={},
            metadata={
                "status": "scaffold",
                "implementation_target": "Tier 3",
                "prerequisites": [
                    "Consolidation RAG Evaluation + 2 published",
                    "Distribution shift test set designed and validated",
                    "Mesa-optimization framing reviewed by research team",
                    "Consolidation RAG runtime integration complete",
                ],
                "planned_ood_shift_type": self.ood_shift_type,
                "research_question": (
                    "Did Consolidation RAG optimize a proxy objective instead of the "
                    "intended retrieval accuracy objective during consolidation? "
                    "Does performance degrade under distribution shift?"
                ),
                "theoretical_framework": "Mesa-optimization (Hubinger et al., 2019)",
                "prior_work": [
                    "Hubinger et al. (2019) — Risks from Learned Optimization",
                    "Langosco et al. (2022) — Goal Misgeneralization in Deep RL",
                    "Shah et al. (2022) — Goal Misgeneralization",
                ],
                "target_venues": [
                    "NeurIPS Safety Workshop",
                    "ICML Alignment Forum",
                    "ARC/METR venues",
                ],
            },
        )
