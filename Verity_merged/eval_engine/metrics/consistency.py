"""
eval_engine/metrics/consistency.py

Consistency / Stability Evaluation

STATUS: Implemented (Tier 2).

RESEARCH QUESTION:
    Does the RAG system produce consistent answers across multiple runs of the
    same query? Does post-consolidation stability improve or degrade compared
    to pre-consolidation baseline?

METRICS:
    - Answer Drift Score (primary): mean pairwise cosine similarity across
      N answers to the same query. 1.0 = identical, 0.0 = completely different.
    - Score Variance: std deviation of RAGAS/NDCG scores across runs.
    - Verdict Drift Rate: fraction of runs where judge verdict changes
      (requires Phase 3 DebateRound integration).
    - Consolidation Stability Index: ratio of post vs pre consolidation
      answer drift scores. >1.0 = consolidation improved consistency.

IMPLEMENTATION:
    Uses sentence-transformers (all-MiniLM-L6-v2) for answer embedding.
    Pairwise cosine similarity computed across all N(N-1)/2 answer pairs.
    Falls back to simple token overlap Jaccard similarity if
    sentence-transformers is not installed.

INSTALL:
    pip install sentence-transformers

PRIOR WORK:
    - BERTScore (Zhang et al., 2020) — semantic similarity via BERT embeddings
    - SelfCheckGPT (Manakul et al., 2023) — consistency via sampling
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity backends
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


def _embed_sentences(texts: list[str], model_name: str) -> np.ndarray:
    """
    Embed texts using sentence-transformers.
    Lazy import — fails gracefully if not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True)
    except ImportError:
        raise ImportError(
            "sentence-transformers required for ConsistencyMetric in 'embedding' mode. "
            "Install with: pip install sentence-transformers\n"
            "Or use mode='jaccard' for a dependency-free fallback."
        )


def _jaccard_similarity(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity as dependency-free fallback.
    Less semantically accurate than cosine similarity but requires no model.
    """
    tokens_a = set(re.findall(r'\b\w+\b', a.lower()))
    tokens_b = set(re.findall(r'\b\w+\b', b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _mean_pairwise_similarity(texts: list[str], model_name: str, mode: str) -> float:
    """
    Compute mean pairwise similarity across all text pairs.

    Args:
        texts:      List of answer strings to compare
        model_name: sentence-transformers model name (embedding mode only)
        mode:       'embedding' | 'jaccard'

    Returns:
        float in [0, 1] — higher = more consistent
    """
    if len(texts) < 2:
        return 1.0  # Single answer is trivially consistent

    if mode == "embedding":
        embeddings = _embed_sentences(texts, model_name)
        sim_matrix = _cosine_similarity_matrix(embeddings)
        n = len(texts)
        # Extract upper triangle (excluding diagonal)
        sims = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(sims))

    elif mode == "jaccard":
        n = len(texts)
        sims = [_jaccard_similarity(texts[i], texts[j])
                for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(sims))

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'embedding' or 'jaccard'.")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """Detailed consistency analysis across N runs."""
    n_runs: int
    answers: list[str]
    answer_drift_score: float          # Mean pairwise similarity (primary)
    score_variance: float              # Std dev of numeric scores across runs (if provided)
    verdict_drift_rate: float          # Fraction of runs with different verdict (if provided)
    consolidation_stability_index: float | None  # post/pre drift ratio (if both provided)
    mode: str                          # 'embedding' | 'jaccard'
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "answer_drift_score": round(self.answer_drift_score, 4),
            "score_variance": round(self.score_variance, 6),
            "verdict_drift_rate": round(self.verdict_drift_rate, 4),
            "consolidation_stability_index": (
                round(self.consolidation_stability_index, 4)
                if self.consolidation_stability_index is not None else None
            ),
            "mode": self.mode,
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class ConsistencyMetric(BaseMetric):
    """
    Consistency / Stability Evaluation.

    Measures how consistently a RAG system answers the same query
    across multiple runs. Useful for detecting instability introduced
    by consolidation cycles or non-deterministic retrieval.

    Two similarity modes:
        'embedding': sentence-transformers cosine similarity (recommended)
        'jaccard':   token overlap (no dependencies, less accurate)

    Usage:
        metric = ConsistencyMetric(n_runs=5, mode='embedding')

        # Single-query mode: pass one answer, score accumulates
        # across multiple score() calls for the same query,
        # then call compute_consistency_report() when done.

        # Multi-run mode (recommended): pass all N answers at once
        result = metric.score_multiple_runs(
            question="What does the consolidation phase improve?",
            answers=[
                "Consolidation improves retrieval precision.",
                "The consolidation phase improves retrieval quality.",
                "Consolidation reduces noise in the knowledge graph.",
                "It improves retrieval by pruning redundant connections.",
                "Consolidation enhances the graph structure for retrieval.",
            ]
        )
        print(result.score)   # mean pairwise similarity
    """

    name = "consistency"

    def __init__(
        self,
        n_runs: int = 5,
        similarity_model: str = "all-MiniLM-L6-v2",
        mode: str = "embedding",
    ) -> None:
        """
        Args:
            n_runs:            Target number of runs for consistency evaluation.
            similarity_model:  sentence-transformers model name (embedding mode).
            mode:              'embedding' (semantic) | 'jaccard' (token overlap).
        """
        self.n_runs = n_runs
        self.similarity_model = similarity_model
        self.mode = mode
        # Internal buffer for single-query accumulation mode
        self._answer_buffer: list[str] = []
        self._score_buffer: list[float] = []
        self._verdict_buffer: list[str] = []
        self._current_question: str = ""

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        numeric_score: float | None = None,
        verdict: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Single-answer accumulation mode.

        Call this once per run for the same query. After n_runs calls,
        call compute_consistency_report() to get the full analysis.

        Args:
            question:      The query (must be same across all runs)
            contexts:      Retrieved contexts (not used in scoring)
            answer:        Generated answer for this run
            numeric_score: Optional metric score for this run (e.g. NDCG@10)
            verdict:       Optional judge verdict for this run (Pass/Conditional/Fail)
        """
        self.validate_inputs(question, contexts, answer)

        # Reset buffer if question changed
        if question != self._current_question:
            self._answer_buffer.clear()
            self._score_buffer.clear()
            self._verdict_buffer.clear()
            self._current_question = question

        self._answer_buffer.append(answer)
        if numeric_score is not None:
            self._score_buffer.append(numeric_score)
        if verdict is not None:
            self._verdict_buffer.append(verdict)

        n = len(self._answer_buffer)
        logger.debug(f"[consistency] Accumulated {n}/{self.n_runs} runs for query")

        # Return intermediate result — call compute_consistency_report() when done
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            raw={},
            metadata={
                "accumulated_runs": n,
                "target_runs": self.n_runs,
                "note": f"Call compute_consistency_report() after {self.n_runs} runs.",
            },
        )

    def score_multiple_runs(
        self,
        question: str,
        answers: list[str],
        numeric_scores: list[float] | None = None,
        verdicts: list[str] | None = None,
        pre_consolidation_answers: list[str] | None = None,
    ) -> MetricResult:
        """
        Preferred mode — pass all N answers at once.

        Args:
            question:                    The query string
            answers:                     N answers from N runs
            numeric_scores:              Optional per-run metric scores
            verdicts:                    Optional per-run judge verdicts
            pre_consolidation_answers:   Pre-consolidation answers for
                                         Consolidation Stability Index computation

        Returns:
            MetricResult with score = answer_drift_score (mean pairwise similarity)
        """
        if not answers:
            return self.error_result("answers list is empty")
        if len(answers) < 2:
            logger.warning("[consistency] Only 1 answer provided — consistency is trivially 1.0")

        try:
            drift_score = _mean_pairwise_similarity(answers, self.similarity_model, self.mode)
        except ImportError as e:
            logger.warning(f"[consistency] Falling back to jaccard: {e}")
            drift_score = _mean_pairwise_similarity(answers, self.similarity_model, "jaccard")
            self.mode = "jaccard"

        # Score variance
        score_var = float(np.var(numeric_scores)) if numeric_scores else 0.0

        # Verdict drift rate
        verdict_drift = 0.0
        if verdicts and len(verdicts) > 1:
            mode_verdict = max(set(verdicts), key=verdicts.count)
            drifted = sum(1 for v in verdicts if v != mode_verdict)
            verdict_drift = drifted / len(verdicts)

        # Consolidation Stability Index
        csi = None
        if pre_consolidation_answers and len(pre_consolidation_answers) >= 2:
            try:
                pre_drift = _mean_pairwise_similarity(
                    pre_consolidation_answers, self.similarity_model, self.mode
                )
                # CSI > 1.0 means post-consolidation is MORE consistent
                csi = drift_score / pre_drift if pre_drift > 0 else None
            except Exception as e:
                logger.warning(f"[consistency] CSI computation failed: {e}")

        result = ConsistencyResult(
            n_runs=len(answers),
            answers=answers,
            answer_drift_score=drift_score,
            score_variance=score_var,
            verdict_drift_rate=verdict_drift,
            consolidation_stability_index=csi,
            mode=self.mode,
            model_used=self.similarity_model if self.mode == "embedding" else "jaccard",
        )

        logger.info(
            f"[consistency] drift_score={drift_score:.4f} | "
            f"score_var={score_var:.4f} | "
            f"verdict_drift={verdict_drift:.1%} | "
            f"csi={round(csi, 3) if csi is not None else 'N/A'} | "
            f"mode={self.mode}"
        )

        return MetricResult(
            metric_name=self.name,
            score=drift_score,
            raw=result.to_dict(),
            metadata={
                "mode": self.mode,
                "n_runs": len(answers),
                "has_pre_consolidation": pre_consolidation_answers is not None,
            },
        )

    def compute_consistency_report(self) -> MetricResult:
        """
        Compute consistency from accumulated single-query calls.
        Call after n_runs calls to score().
        """
        if not self._answer_buffer:
            return self.error_result(
                "No answers accumulated. Call score() for each run first, "
                "or use score_multiple_runs() directly."
            )
        result = self.score_multiple_runs(
            question=self._current_question,
            answers=self._answer_buffer,
            numeric_scores=self._score_buffer or None,
            verdicts=self._verdict_buffer or None,
        )
        self._answer_buffer.clear()
        self._score_buffer.clear()
        self._verdict_buffer.clear()
        return result

    def reset(self) -> None:
        """Clear accumulated buffer."""
        self._answer_buffer.clear()
        self._score_buffer.clear()
        self._verdict_buffer.clear()
        self._current_question = ""

    async def score_async(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Async wrapper — embedding is synchronous; runs in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.score(question, contexts, answer, ground_truth, **kwargs),
        )
