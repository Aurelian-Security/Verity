"""
eval_engine/metrics/query_perturbation.py

Consolidation RAG Evaluation — Test 5: Query Perturbation Test

PURPOSE:
    Measures Consolidation RAG's retrieval robustness under query-side perturbations.
    Tests whether the post-consolidation knowledge graph remains queryable
    when user queries are noisy, paraphrased, or adversarially modified.

    This is distinct from poisoning (Test 4):
        - Poisoning: adversary modifies the CORPUS
        - Perturbation: adversary/noise modifies the QUERY

    For Consolidation RAG specifically, the question is whether the consolidation
    cycle's graph structure is robust to semantic drift in query phrasing —
    i.e., does REM-phase pattern synthesis create retrieval pathways that
    tolerate query variation?

PERTURBATION TYPES:
    1. Paraphrase:    Semantically equivalent restatement of the query
    2. Typo/noise:    Character-level perturbations (misspellings, swaps)
    3. Truncation:    Query shortened to partial form
    4. Expansion:     Query padded with irrelevant context
    5. Adversarial:   TextAttack-style perturbations designed to maximally
                      degrade retrieval while preserving surface meaning

TOOLS:
    - TextAttack (pip install textattack) for adversarial perturbations
    - Simple string ops for typo, truncation, expansion
    - Paraphrase via LLM call (optional, adds cost)

METRICS:
    Primary:
        - Perturbation Robustness Score (PRS): mean NDCG@K ratio
          across perturbation types (perturbed_score / clean_score)
          PRS = 1.0 → fully robust, PRS = 0.0 → completely degraded
    Secondary:
        - Score variance across perturbation types
        - Per-perturbation-type NDCG@K breakdown

PAPER 1 SCOPE:
    Types 1-4 (paraphrase, typo, truncation, expansion) in scope.
    Type 5 (adversarial TextAttack) deferred to Longitudinal Eval Suite/derivative pipeline.
"""

from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.metrics.retrieval_metrics import ndcg_at_k
from eval_engine.schemas import RetrievalCase, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturbation types
# ---------------------------------------------------------------------------

class PerturbationType:
    PARAPHRASE = "paraphrase"
    TYPO = "typo_noise"
    TRUNCATION = "truncation"
    EXPANSION = "expansion"
    ADVERSARIAL = "adversarial"   # Longitudinal Eval Suite scope


# ---------------------------------------------------------------------------
# Built-in perturbation functions
# ---------------------------------------------------------------------------

def _typo_noise(query: str, noise_rate: float = 0.1, seed: int = 42) -> str:
    """
    Introduce character-level noise at `noise_rate` fraction of characters.
    Operations: swap adjacent chars, delete char, insert random char.
    """
    rng = random.Random(seed)
    chars = list(query)
    n_ops = max(1, int(len(chars) * noise_rate))

    for _ in range(n_ops):
        if not chars:
            break
        op = rng.choice(["swap", "delete", "insert"])
        idx = rng.randint(0, len(chars) - 1)

        if op == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "delete":
            chars.pop(idx)
        elif op == "insert":
            chars.insert(idx, rng.choice(string.ascii_lowercase))

    return "".join(chars)


def _truncate(query: str, keep_fraction: float = 0.6) -> str:
    """Keep the first `keep_fraction` of words."""
    words = query.split()
    keep = max(1, int(len(words) * keep_fraction))
    return " ".join(words[:keep])


def _expand(query: str, padding: str = "Please provide detailed information about") -> str:
    """Prepend irrelevant context to the query."""
    return f"{padding} {query}"


def _paraphrase_simple(query: str) -> str:
    """
    Simple rule-based paraphrase for testing without LLM cost.
    Replaces common question starters.
    For Consolidation Eval Suite: use LLM-based paraphrase via paraphrase_with_llm().
    """
    replacements = [
        ("What is", "Can you explain"),
        ("How does", "In what way does"),
        ("Why does", "What is the reason"),
        ("When did", "At what point did"),
        ("Where is", "What is the location of"),
        ("Who is", "Can you identify"),
    ]
    result = query
    for original, replacement in replacements:
        if result.startswith(original):
            result = replacement + result[len(original):]
            break
    return result


async def paraphrase_with_llm(query: str, model: str = "claude-haiku-4-5") -> str:
    """
    LLM-based paraphrase. Uses Claude Haiku (low cost).
    Call this for Consolidation Eval Suite paraphrase perturbations.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": (
                    f"Paraphrase the following query in different words, "
                    f"preserving the exact same meaning. Return only the "
                    f"paraphrased query, nothing else.\n\nQuery: {query}"
                ),
            }],
        )
        return message.content[0].text.strip()
    except Exception as e:
        logger.warning(f"LLM paraphrase failed: {e}. Falling back to rule-based.")
        return _paraphrase_simple(query)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PerturbationTypeResult:
    """Results for one perturbation type across all queries."""
    perturbation_type: str
    mean_ndcg_clean: float
    mean_ndcg_perturbed: float
    robustness_score: float         # perturbed / clean (1.0 = fully robust)
    score_variance: float
    n_queries: int
    per_query_scores: list[float] = field(default_factory=list)


@dataclass
class PerturbationReport:
    """Aggregated query perturbation test results."""
    experiment_id: str
    k: int
    type_results: list[PerturbationTypeResult]

    @property
    def overall_robustness_score(self) -> float:
        """Mean PRS across all perturbation types."""
        if not self.type_results:
            return 0.0
        return float(np.mean([t.robustness_score for t in self.type_results]))

    @property
    def most_vulnerable_type(self) -> str | None:
        if not self.type_results:
            return None
        return min(self.type_results, key=lambda t: t.robustness_score).perturbation_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "k": self.k,
            "overall_robustness_score": round(self.overall_robustness_score, 4),
            "most_vulnerable_type": self.most_vulnerable_type,
            "type_results": [
                {
                    "type": t.perturbation_type,
                    "mean_ndcg_clean": round(t.mean_ndcg_clean, 4),
                    "mean_ndcg_perturbed": round(t.mean_ndcg_perturbed, 4),
                    "robustness_score": round(t.robustness_score, 4),
                    "score_variance": round(t.score_variance, 6),
                    "n_queries": t.n_queries,
                }
                for t in self.type_results
            ],
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  PERTURBATION REPORT: {self.experiment_id}")
        print(f"  Overall Robustness Score (PRS): {self.overall_robustness_score:.4f}")
        print(f"  Most vulnerable type: {self.most_vulnerable_type}")
        print(f"{'='*60}")
        for t in sorted(self.type_results, key=lambda x: x.robustness_score):
            bar = "█" * int(t.robustness_score * 20)
            print(
                f"  {t.perturbation_type:<22} PRS={t.robustness_score:.3f} "
                f"[{bar:<20}] "
                f"(clean={t.mean_ndcg_clean:.3f}, perturbed={t.mean_ndcg_perturbed:.3f})"
            )
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class QueryPerturbationMetric(BaseMetric):
    """
    Consolidation RAG Evaluation Test 5: Query Perturbation.

    Measures retrieval robustness under query-side perturbations.

    Usage:
        metric = QueryPerturbationMetric(k=10)
        report = metric.run_perturbation_test(
            cases=eval_cases,
            retriever=consolidation_rag.retrieve,
            relevance_by_query=relevance_labels,
            perturbation_types=[
                PerturbationType.TYPO,
                PerturbationType.TRUNCATION,
                PerturbationType.PARAPHRASE,
                PerturbationType.EXPANSION,
            ],
        )
    """

    name = "query_perturbation"

    def __init__(
        self,
        k: int = 10,
        typo_noise_rate: float = 0.1,
        truncation_keep: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.typo_noise_rate = typo_noise_rate
        self.truncation_keep = truncation_keep
        self.seed = seed

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        perturbation_type: str = PerturbationType.TYPO,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Score a single query with a perturbation applied.
        Returns the perturbed query and a flag — actual scoring
        requires a retriever call (use run_perturbation_test for full eval).
        """
        self.validate_inputs(question, contexts, answer)
        perturbed = self._perturb(question, perturbation_type)

        return MetricResult(
            metric_name=self.name,
            score=0.0,  # placeholder — real score requires retriever call
            raw={"original_query": question, "perturbed_query": perturbed},
            metadata={
                "perturbation_type": perturbation_type,
                "note": "Use run_perturbation_test() for full NDCG@K evaluation.",
            },
        )

    def run_perturbation_test(
        self,
        cases: list[RetrievalCase],
        retriever: Callable[[str], list[str]],
        relevance_by_query: dict[str, dict[str, float]],
        perturbation_types: list[str] | None = None,
        experiment_id: str = "perturbation_test",
    ) -> PerturbationReport:
        """
        Run full query perturbation test across all types.

        Args:
            cases:              Benchmark cases
            retriever:          fn(query: str) -> list[retrieved_ids]
            relevance_by_query: Graded relevance labels
            perturbation_types: List of PerturbationType constants
                                Defaults to [TYPO, TRUNCATION, PARAPHRASE, EXPANSION]
            experiment_id:      Run identifier
        """
        if perturbation_types is None:
            perturbation_types = [
                PerturbationType.TYPO,
                PerturbationType.TRUNCATION,
                PerturbationType.PARAPHRASE,
                PerturbationType.EXPANSION,
            ]

        # Clean baseline scores
        clean_results = self._retrieve_all(cases, retriever)
        clean_ndcg_by_query = self._score_per_query(cases, clean_results, relevance_by_query)
        mean_clean = float(np.mean(list(clean_ndcg_by_query.values()))) if clean_ndcg_by_query else 0.0
        logger.info(f"[perturbation] Clean baseline NDCG@{self.k}: {mean_clean:.4f}")

        type_results = []
        for p_type in perturbation_types:
            perturbed_cases = [
                RetrievalCase(
                    query_id=c.query_id,
                    query=self._perturb(c.query, p_type),
                    relevant_ids=c.relevant_ids,
                )
                for c in cases
            ]
            perturbed_results = self._retrieve_all(perturbed_cases, retriever)
            perturbed_ndcg_by_query = self._score_per_query(
                perturbed_cases, perturbed_results, relevance_by_query
            )

            perturbed_scores = list(perturbed_ndcg_by_query.values())
            mean_perturbed = float(np.mean(perturbed_scores)) if perturbed_scores else 0.0
            prs = mean_perturbed / mean_clean if mean_clean > 0 else 0.0

            type_results.append(PerturbationTypeResult(
                perturbation_type=p_type,
                mean_ndcg_clean=mean_clean,
                mean_ndcg_perturbed=mean_perturbed,
                robustness_score=min(prs, 1.0),   # cap at 1.0 (can't exceed clean)
                score_variance=float(np.var(perturbed_scores)) if perturbed_scores else 0.0,
                n_queries=len(cases),
                per_query_scores=perturbed_scores,
            ))
            logger.info(
                f"[perturbation] {p_type}: "
                f"NDCG@{self.k}={mean_perturbed:.4f}, PRS={prs:.4f}"
            )

        report = PerturbationReport(
            experiment_id=experiment_id,
            k=self.k,
            type_results=type_results,
        )
        report.print_summary()
        return report

    def _perturb(self, query: str, p_type: str) -> str:
        """Apply a single perturbation type to a query."""
        if p_type == PerturbationType.TYPO:
            return _typo_noise(query, self.typo_noise_rate, self.seed)
        elif p_type == PerturbationType.TRUNCATION:
            return _truncate(query, self.truncation_keep)
        elif p_type == PerturbationType.EXPANSION:
            return _expand(query)
        elif p_type == PerturbationType.PARAPHRASE:
            return _paraphrase_simple(query)
        elif p_type == PerturbationType.ADVERSARIAL:
            logger.warning(
                "Adversarial perturbation requires TextAttack. "
                "Falling back to typo noise. "
                "Longitudinal Eval Suite scope: install textattack and implement adversarial attack."
            )
            return _typo_noise(query, noise_rate=0.20, seed=self.seed)
        else:
            logger.warning(f"Unknown perturbation type '{p_type}'. Returning original.")
            return query

    def _retrieve_all(
        self,
        cases: list[RetrievalCase],
        retriever: Callable[[str], list[str]],
    ) -> list[RetrievalResult]:
        results = []
        for case in cases:
            try:
                retrieved = retriever(case.query)
                results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=retrieved))
            except Exception as e:
                logger.error(f"Retriever failed for '{case.query_id}': {e}")
                results.append(RetrievalResult(query_id=case.query_id, retrieved_ids=[]))
        return results

    def _score_per_query(
        self,
        cases: list[RetrievalCase],
        results: list[RetrievalResult],
        relevance_by_query: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        scores = {}
        result_map = {r.query_id: r for r in results}
        for case in cases:
            r = result_map.get(case.query_id)
            rel = relevance_by_query.get(case.query_id, {})
            if r and rel:
                scores[case.query_id] = ndcg_at_k({case.query_id: rel}, [r], self.k)
            else:
                scores[case.query_id] = 0.0
        return scores
