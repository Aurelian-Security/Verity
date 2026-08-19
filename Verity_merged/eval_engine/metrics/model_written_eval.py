"""
eval_engine/metrics/model_written_eval.py

Model Written Evaluation (MWE) / Policy Rubric Scoring

STATUS: Implemented (Tier 2).

RESEARCH QUESTION:
    Does the RAG system's output satisfy policy-defined quality rubrics?
    Not "is this factually correct?" but "does this meet the standard
    a governance team or compliance framework would require?"

DISTINCTION FROM OTHER METRICS:
    - Faithfulness:        Is it grounded in retrieved context?
    - Hallucination Rate:  Are specific claims unsupported?
    - Constitutional Eval: Does it satisfy alignment principles?
    - MWE (this module):   Does it satisfy a CALLER-DEFINED policy rubric.
                           The rubric is flexible — encodes domain standards.

IMPLEMENTATION:
    Caller defines rubric as a list of RubricCriterion objects.
    Each criterion has a type:
        'binary':    LLM returns PASS or FAIL → 1.0 or 0.0
        'scored_5':  LLM returns 1-5 → normalized to 0.0-1.0
        'scored_10': LLM returns 1-10 → normalized to 0.0-1.0
    Weighted aggregate returned as primary score.
    Required criteria that fail cause overall score to 0.0.

PRIOR WORK:
    - OpenAI Evals Framework
    - MT-Bench (Zheng et al., 2023) — LLM-as-judge
    - AlpacaEval (Li et al., 2023) — automated instruction-following eval
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rubric criterion
# ---------------------------------------------------------------------------

@dataclass
class RubricCriterion:
    """Single criterion in a policy rubric."""
    name: str
    prompt: str
    type: str = "binary"       # "binary" | "scored_5" | "scored_10"
    weight: float = 1.0
    required: bool = False     # If True, failure on this criterion → overall score = 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RubricResult:
    """Per-criterion scoring results."""
    criterion_scores: dict[str, float]
    criterion_raw: dict[str, str]
    composite_score: float
    failed_required: list[str]
    rubric_passed: bool           # True if no required criteria failed
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 4),
            "rubric_passed": self.rubric_passed,
            "failed_required": self.failed_required,
            "criterion_scores": {k: round(v, 4) for k, v in self.criterion_scores.items()},
            "criterion_raw_responses": self.criterion_raw,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class ModelWrittenEvalMetric(BaseMetric):
    """
    Model Written Evaluation — caller-defined policy rubric scoring.

    Usage:
        # Use the built-in governance rubric
        metric = ModelWrittenEvalMetric.default_governance_rubric()
        result = metric.score(
            question="What is the refund policy?",
            contexts=["Our refund policy allows returns within 30 days..."],
            answer="You can return items within 30 days for a full refund.",
        )
        print(result.score)              # composite rubric score
        print(result.raw["rubric_passed"])  # True/False

        # Or define a custom rubric
        from eval_engine.metrics.model_written_eval import RubricCriterion
        metric = ModelWrittenEvalMetric(rubric=[
            RubricCriterion(
                name="cites_policy",
                prompt="Does the answer reference the specific policy it draws from?",
                type="binary", weight=0.5, required=True,
            ),
            RubricCriterion(
                name="clarity",
                prompt="Rate the clarity of this answer from 1-5.",
                type="scored_5", weight=0.5,
            ),
        ])
    """

    name = "model_written_eval"

    def __init__(
        self,
        rubric: list[RubricCriterion] | None = None,
        judge_model: str = "claude-haiku-4-5",
        dry_run: bool = False,
    ) -> None:
        self.rubric = rubric or []
        self.judge_model = judge_model
        self.dry_run = dry_run
        self._sanitizer = InputSanitizer(strict=False)

        if not self.rubric:
            logger.warning(
                "[model_written_eval] No rubric criteria defined. "
                "Use ModelWrittenEvalMetric.default_governance_rubric() "
                "or pass a list of RubricCriterion objects."
            )

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Evaluate answer against all rubric criteria.

        Returns:
            MetricResult with score = weighted composite rubric score.
            score = 0.0 if any required criterion fails.
            result.raw contains per-criterion scores and pass/fail status.
        """
        self.validate_inputs(question, contexts, answer)

        if not self.rubric:
            return self.error_result(
                "No rubric criteria defined. Pass rubric=[...] at construction "
                "or use ModelWrittenEvalMetric.default_governance_rubric()."
            )

        _, clean_answer, _ = self._sanitizer.sanitize(question, answer, [])

        if self.dry_run:
            return self._dry_run_result()

        criterion_scores: dict[str, float] = {}
        criterion_raw: dict[str, str] = {}
        failed_required: list[str] = []

        # Normalize weights
        total_weight = sum(c.weight for c in self.rubric)
        normalized_weights = {c.name: c.weight / total_weight for c in self.rubric}

        context_block = "\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts[:3]))

        for criterion in self.rubric:
            score_val, raw_response = self._score_criterion(
                criterion=criterion,
                question=question,
                answer=clean_answer,
                context_block=context_block,
            )
            criterion_scores[criterion.name] = score_val
            criterion_raw[criterion.name] = raw_response

            if criterion.required and score_val < 0.5:
                failed_required.append(criterion.name)
                logger.warning(
                    f"[model_written_eval] Required criterion '{criterion.name}' "
                    f"failed (score={score_val:.3f})"
                )

        # Compute composite
        composite = sum(
            criterion_scores[c.name] * normalized_weights[c.name]
            for c in self.rubric
        )

        # Required failure overrides composite
        rubric_passed = len(failed_required) == 0
        if not rubric_passed:
            composite = 0.0
            logger.warning(
                f"[model_written_eval] Rubric FAILED — required criteria: {failed_required}. "
                f"Composite score set to 0.0."
            )

        result = RubricResult(
            criterion_scores=criterion_scores,
            criterion_raw=criterion_raw,
            composite_score=composite,
            failed_required=failed_required,
            rubric_passed=rubric_passed,
        )

        logger.info(
            f"[model_written_eval] composite={composite:.4f} | "
            f"passed={rubric_passed} | criteria={criterion_scores}"
        )

        return MetricResult(
            metric_name=self.name,
            score=composite,
            raw=result.to_dict(),
            metadata={
                "judge_model": self.judge_model,
                "n_criteria": len(self.rubric),
                "rubric_passed": rubric_passed,
                "failed_required": failed_required,
            },
        )

    def _score_criterion(
        self,
        criterion: RubricCriterion,
        question: str,
        answer: str,
        context_block: str,
    ) -> tuple[float, str]:
        """Score a single rubric criterion via LLM judge."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            if criterion.type == "binary":
                instruction = (
                    f"{criterion.prompt}\n"
                    f"Reply with ONLY one word: PASS or FAIL."
                )
                max_tokens = 10
            elif criterion.type == "scored_5":
                instruction = (
                    f"{criterion.prompt}\n"
                    f"Reply with ONLY a single integer from 1 to 5."
                )
                max_tokens = 5
            elif criterion.type == "scored_10":
                instruction = (
                    f"{criterion.prompt}\n"
                    f"Reply with ONLY a single integer from 1 to 10."
                )
                max_tokens = 5
            else:
                instruction = f"{criterion.prompt}\nReply with ONLY a float between 0.0 and 1.0."
                max_tokens = 10

            prompt = (
                f"Question: {question}\n\n"
                f"Context:\n{context_block}\n\n"
                f"Answer to evaluate: {answer}\n\n"
                f"Criterion — {criterion.name.upper()}:\n{instruction}"
            )

            message = client.messages.create(
                model=self.judge_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip().upper()

            # Parse response based on criterion type
            if criterion.type == "binary":
                score = 1.0 if "PASS" in raw else 0.0
            elif criterion.type == "scored_5":
                matches = re.findall(r'\d+', raw)
                if matches:
                    val = int(matches[0])
                    score = max(0.0, min(1.0, (val - 1) / 4.0))
                else:
                    score = 0.5
            elif criterion.type == "scored_10":
                matches = re.findall(r'\d+', raw)
                if matches:
                    val = int(matches[0])
                    score = max(0.0, min(1.0, (val - 1) / 9.0))
                else:
                    score = 0.5
            else:
                matches = re.findall(r"[0-9]*\.?[0-9]+", raw)
                score = float(matches[0]) if matches else 0.5
                score = max(0.0, min(1.0, score))

            return score, raw

        except ImportError:
            raise ImportError("anthropic SDK required: pip install anthropic")
        except Exception as e:
            logger.error(f"[model_written_eval] Criterion '{criterion.name}' failed: {e}")
            return 0.5, str(e)

    def _dry_run_result(self) -> MetricResult:
        """Return mock scores without API calls."""
        criterion_scores = {c.name: 0.85 for c in self.rubric}
        composite = 0.85
        result = RubricResult(
            criterion_scores=criterion_scores,
            criterion_raw={c.name: "DRY-RUN" for c in self.rubric},
            composite_score=composite,
            failed_required=[],
            rubric_passed=True,
            dry_run=True,
        )
        return MetricResult(
            metric_name=self.name,
            score=composite,
            raw=result.to_dict(),
            metadata={"judge_model": f"{self.judge_model}[DRY-RUN]", "dry_run": True},
        )

    async def score_async(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Async wrapper — LLM calls run in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.score(question, contexts, answer, ground_truth, **kwargs),
        )

    @classmethod
    def default_governance_rubric(
        cls,
        judge_model: str = "claude-haiku-4-5",
        dry_run: bool = False,
    ) -> "ModelWrittenEvalMetric":
        """
        Pre-built rubric for RAG governance evaluation.

        Criteria:
            source_attribution    (binary, 0.25 weight)
            uncertainty_disclosure (binary, 0.25 weight)
            scope_adherence       (binary, 0.30 weight, REQUIRED)
            response_quality      (scored_5, 0.20 weight)
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
                prompt=(
                    "Does the answer stay within the scope of the retrieved context "
                    "without introducing outside claims not supported by the context?"
                ),
                type="binary", weight=0.30, required=True,
            ),
            RubricCriterion(
                name="response_quality",
                prompt="Rate the overall clarity and usefulness of this response from 1-5.",
                type="scored_5", weight=0.20, required=False,
            ),
        ]
        return cls(rubric=rubric, judge_model=judge_model, dry_run=dry_run)

    @classmethod
    def research_paper_rubric(
        cls,
        judge_model: str = "claude-sonnet-4-6",
        dry_run: bool = False,
    ) -> "ModelWrittenEvalMetric":
        """
        Rubric for evaluating RAG outputs against research paper quality standards.
        Uses Sonnet for higher reasoning quality.
        """
        rubric = [
            RubricCriterion(
                name="factual_accuracy",
                prompt="Is the answer factually accurate based on the retrieved context?",
                type="binary", weight=0.35, required=True,
            ),
            RubricCriterion(
                name="completeness",
                prompt="Does the answer address all key aspects of the question?",
                type="scored_5", weight=0.25, required=False,
            ),
            RubricCriterion(
                name="citation_quality",
                prompt="Does the answer properly attribute claims to specific context sources?",
                type="binary", weight=0.20, required=False,
            ),
            RubricCriterion(
                name="precision",
                prompt="Is the answer free of unnecessary hedging or vague language?",
                type="scored_5", weight=0.20, required=False,
            ),
        ]
        return cls(rubric=rubric, judge_model=judge_model, dry_run=dry_run)
