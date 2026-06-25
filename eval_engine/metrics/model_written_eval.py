"""
eval_engine/metrics/model_written_eval.py

Model Written Evaluation (MWE) / Rubric Scoring — SCAFFOLD

STATUS: Scaffold. Interface locked. Implementation deferred to Tier 2.

RESEARCH QUESTION:
    Does Consolidation RAG's output satisfy policy-defined quality rubrics?
    Not "is this factually correct?" but "does this meet the standard
    a governance team or compliance framework would require?"

    Analogous to OpenAI's evals framework and Anthropic's internal
    model-written evaluation approach.

DISTINCTION FROM OTHER METRICS:
    - Faithfulness:        Is it grounded in retrieved context?
    - Hallucination Rate:  Are specific claims unsupported?
    - Constitutional Eval: Does it satisfy alignment principles?
    - MWE (this module):   Does it satisfy a custom policy rubric?
                           The rubric is caller-defined, not fixed.

    MWE is the most flexible evaluation type — it can encode:
        - Domain-specific quality standards (medical, legal, financial)
        - Organizational policy requirements
        - Safety governance checklists
        - Research paper review criteria

PLANNED IMPLEMENTATION:
    1. Caller defines a rubric as a list of criteria with pass/fail
       or scored (0-5) dimensions
    2. LLM judge evaluates the answer against each criterion
    3. Results aggregated into overall rubric score
    4. Individual criterion scores logged for auditability

EXAMPLE RUBRIC (for Consolidation RAG governance use case):
    criteria = [
        {"name": "source_attribution", "type": "binary",
         "prompt": "Does the answer cite its sources?"},
        {"name": "uncertainty_disclosure", "type": "binary",
         "prompt": "Does the answer acknowledge what it doesn't know?"},
        {"name": "response_quality", "type": "scored_5",
         "prompt": "Rate the overall response quality from 1-5."},
    ]

INTEGRATION:
    - Governance and compliance reporting layer
    - Attaches to JudgeAgent output in Phase 3
    - Feeds into TrustScore as optional component via custom weight

PAPER PLACEMENT:
    Extended Eval Suite or 4 — "Policy-Aligned Evaluation for RAG Governance"
    Strong fit for IEEE Security & Privacy, FAccT, or AIES venues.

PRIOR WORK TO CITE:
    - OpenAI Evals (Owain Evans et al.) — model-written evaluation framework
    - MT-Bench (Zheng et al., 2023) — LLM-as-judge evaluation
    - AlpacaEval (Li et al., 2023) — automated instruction-following eval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """Single criterion in a policy rubric."""
    name: str
    prompt: str
    type: str = "binary"      # "binary" | "scored_5" | "scored_10"
    weight: float = 1.0
    required: bool = False    # If True, failure on this criterion fails the whole rubric


class ModelWrittenEvalMetric(BaseMetric):
    """
    Model Written Evaluation — Policy rubric scoring.

    SCAFFOLD — returns NotImplemented result with research context.
    Implementation target: Tier 2.
    """

    name = "model_written_eval"

    def __init__(
        self,
        rubric: list[RubricCriterion] | None = None,
        judge_model: str = "claude-sonnet-4-6",
    ) -> None:
        self.rubric = rubric or []
        self.judge_model = judge_model

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

        When implemented, evaluates answer against each rubric criterion
        using an LLM judge, returns weighted rubric score.
        """
        logger.info(
            "[model_written_eval] SCAFFOLD — implementation deferred to Tier 2."
        )
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            raw={},
            metadata={
                "status": "scaffold",
                "implementation_target": "Tier 2",
                "n_criteria": len(self.rubric),
                "planned_judge_model": self.judge_model,
                "research_question": (
                    "Does Consolidation RAG's output satisfy policy-defined quality rubrics? "
                    "Rubric is caller-defined — encodes domain or governance standards."
                ),
                "prior_work": [
                    "OpenAI Evals Framework",
                    "MT-Bench (Zheng et al., 2023)",
                    "AlpacaEval (Li et al., 2023)",
                ],
            },
        )

    @classmethod
    def default_governance_rubric(cls) -> "ModelWrittenEvalMetric":
        """
        Pre-built rubric for RAG governance evaluation.
        Returns a configured instance with standard governance criteria.
        Ready to use once implementation is complete.
        """
        rubric = [
            RubricCriterion(
                name="source_attribution",
                prompt="Does the answer cite or reference the sources it draws from?",
                type="binary", weight=0.25, required=False,
            ),
            RubricCriterion(
                name="uncertainty_disclosure",
                prompt="Does the answer acknowledge gaps or uncertainty in its knowledge?",
                type="binary", weight=0.25, required=False,
            ),
            RubricCriterion(
                name="scope_adherence",
                prompt="Does the answer stay within the scope of the retrieved context "
                       "without introducing outside claims?",
                type="binary", weight=0.30, required=True,
            ),
            RubricCriterion(
                name="response_quality",
                prompt="Rate the overall clarity and usefulness of this response from 1-5.",
                type="scored_5", weight=0.20, required=False,
            ),
        ]
        return cls(rubric=rubric)
