"""
eval_engine/metrics/source_reliability.py

Source Reliability Scoring

STATUS: Implemented (Tier 2).
PREREQUISITE MET: SourceMetadata added to schemas.py, RetrievalResult extended.

RESEARCH QUESTION:
    How reliable are the sources that the RAG system retrieves?
    Does post-consolidation retrieval surface more reliable sources
    than pre-consolidation baseline?

DIMENSIONS:
    1. Authority Score (0.0–1.0)
       Domain-based authority: peer-reviewed > institutional > general web.
    2. Consistency Score (0.0–1.0)
       Cross-source agreement — do retrieved sources agree with each other?
       Uses LLM judge to assess claim-level agreement.
    3. Citation Quality (0.0–1.0)
       Citation count as proxy for community validation.
       Decays logarithmically — 100 citations is not 10x better than 10.
    4. Recency Score (0.0–1.0)
       Exponential decay based on days since publication.
       Half-life configurable (default: 365 days).

USAGE:
    from eval_engine.schemas import RetrievalResult, SourceMetadata

    result = RetrievalResult(
        query_id="q1",
        retrieved_ids=["doc_1", "doc_2"],
        source_metadata=[
            SourceMetadata(
                source_id="doc_1",
                domain="arxiv.org",
                publication_date="2024-01-15",
                citation_count=42,
                is_peer_reviewed=True,
            ),
            SourceMetadata(
                source_id="doc_2",
                domain="wikipedia.org",
                publication_date="2023-06-01",
                citation_count=0,
                is_peer_reviewed=False,
            ),
        ]
    )

    metric = SourceReliabilityMetric()
    score = metric.score_from_result(result, question="...", contexts=[...])

PRIOR WORK:
    - FEVER (Thorne et al., 2018) — fact verification methodology
    - MediaBias/FactCheck reliability taxonomy
    - NewsGuard credibility scoring
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.schemas import RetrievalResult, SourceMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authority domain taxonomy
# ---------------------------------------------------------------------------

# Higher score = more authoritative
DOMAIN_AUTHORITY: dict[str, float] = {
    # Tier 1 — Peer-reviewed / academic
    "arxiv.org": 0.85,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "scholar.google.com": 0.80,
    "doi.org": 0.90,
    "acm.org": 0.90,
    "ieee.org": 0.90,
    "nature.com": 0.95,
    "science.org": 0.95,
    "springer.com": 0.85,
    "wiley.com": 0.85,
    "oxford.ac.uk": 0.90,
    "cambridge.org": 0.90,
    # Tier 2 — Government / institutional
    "gov": 0.85,
    "edu": 0.80,
    "nist.gov": 0.90,
    "cdc.gov": 0.90,
    "who.int": 0.90,
    # Tier 3 — Reference / encyclopedia
    "wikipedia.org": 0.65,
    "britannica.com": 0.70,
    # Tier 4 — General web (default)
    "default": 0.40,
}


def _authority_score(source: SourceMetadata) -> float:
    """Compute authority score from domain and peer-review status."""
    base = 0.40

    if source.domain:
        domain_lower = source.domain.lower().strip("www.")
        # Exact match first
        if domain_lower in DOMAIN_AUTHORITY:
            base = DOMAIN_AUTHORITY[domain_lower]
        else:
            # Check TLD suffix
            for pattern, score in DOMAIN_AUTHORITY.items():
                if domain_lower.endswith(f".{pattern}") or domain_lower == pattern:
                    base = score
                    break

    # Peer review bonus
    if source.is_peer_reviewed:
        base = min(1.0, base + 0.10)

    return base


def _citation_score(citation_count: int | None, max_citations: int = 1000) -> float:
    """
    Logarithmic citation score. 0 citations → 0.0, max_citations → 1.0.
    Logarithm prevents linear dominance by highly-cited sources.
    """
    if citation_count is None or citation_count <= 0:
        return 0.0
    return min(1.0, math.log(citation_count + 1) / math.log(max_citations + 1))


def _recency_score(publication_date: str | None, half_life_days: int = 365) -> float:
    """
    Exponential decay recency score.
    Recent publication → high score. Older → lower.
    half_life_days: days until score halves (default: 1 year).
    """
    if not publication_date:
        return 0.5  # Unknown date → neutral

    try:
        # Try parsing ISO date
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                pub_date = datetime.strptime(publication_date, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            return 0.5

        now = datetime.now(tz=timezone.utc)
        days_old = (now - pub_date).days
        if days_old < 0:
            return 1.0  # Future date (data error) → treat as very recent
        return math.exp(-days_old * math.log(2) / half_life_days)

    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SourceReliabilityResult:
    """Per-source reliability scores."""
    source_scores: dict[str, dict[str, float]]   # source_id → dimension scores
    mean_authority: float
    mean_citation: float
    mean_recency: float
    consistency_score: float
    composite_score: float
    n_sources: int
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 4),
            "n_sources": self.n_sources,
            "mean_authority": round(self.mean_authority, 4),
            "mean_citation": round(self.mean_citation, 4),
            "mean_recency": round(self.mean_recency, 4),
            "consistency_score": round(self.consistency_score, 4),
            "weights": self.weights,
            "source_scores": {
                sid: {k: round(v, 4) for k, v in scores.items()}
                for sid, scores in self.source_scores.items()
            },
        }


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class SourceReliabilityMetric(BaseMetric):
    """
    Source Reliability Scoring.

    Requires SourceMetadata attached to RetrievalResult.
    Use score_from_result() for the primary interface.

    Usage:
        metric = SourceReliabilityMetric()
        result = metric.score_from_result(
            retrieval_result=retrieval_result,  # must have source_metadata
            question="What is the safety profile of X?",
            contexts=retrieved_text_chunks,
        )
        print(result.score)                          # composite reliability
        print(result.raw["mean_authority"])          # mean authority score
        print(result.raw["source_scores"])           # per-source breakdown
    """

    name = "source_reliability"

    def __init__(
        self,
        authority_weight: float = 0.35,
        consistency_weight: float = 0.30,
        citation_weight: float = 0.20,
        recency_weight: float = 0.15,
        recency_decay_days: int = 365,
        citation_max: int = 1000,
        check_consistency: bool = False,   # LLM-based; adds cost. Off by default.
        judge_model: str = "claude-haiku-4-5",
    ) -> None:
        total = authority_weight + consistency_weight + citation_weight + recency_weight
        self.weights = {
            "authority": authority_weight / total,
            "consistency": consistency_weight / total,
            "citation": citation_weight / total,
            "recency": recency_weight / total,
        }
        self.recency_decay_days = recency_decay_days
        self.citation_max = citation_max
        self.check_consistency = check_consistency
        self.judge_model = judge_model

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        source_metadata: list[SourceMetadata] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Score source reliability from SourceMetadata kwarg.

        Pass source_metadata as a kwarg when calling via EvalRunner.
        For typed usage with RetrievalResult, prefer score_from_result().
        """
        self.validate_inputs(question, contexts, answer)

        if not source_metadata:
            logger.warning(
                "[source_reliability] No source_metadata provided. "
                "Cannot compute reliability scores. "
                "Pass source_metadata=[SourceMetadata(...), ...] as a kwarg, "
                "or use score_from_result() with a RetrievalResult."
            )
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw={"note": "No source metadata provided"},
                metadata={"has_source_metadata": False},
            )

        return self._compute(question, contexts, source_metadata)

    def score_from_result(
        self,
        retrieval_result: RetrievalResult,
        question: str,
        contexts: list[str],
    ) -> MetricResult:
        """
        Primary interface — score from a typed RetrievalResult.

        Requires retrieval_result.source_metadata to be populated.
        """
        if not retrieval_result.source_metadata:
            return self.error_result(
                "RetrievalResult.source_metadata is None. "
                "Attach SourceMetadata when constructing RetrievalResult."
            )
        return self._compute(question, contexts, retrieval_result.source_metadata)

    def _compute(
        self,
        question: str,
        contexts: list[str],
        sources: list[SourceMetadata],
    ) -> MetricResult:
        """Core scoring logic."""
        if not sources:
            return self.error_result("source_metadata list is empty")

        source_scores: dict[str, dict[str, float]] = {}
        authority_scores, citation_scores, recency_scores = [], [], []

        for source in sources:
            auth = _authority_score(source)
            cit = _citation_score(source.citation_count, self.citation_max)
            rec = _recency_score(source.publication_date, self.recency_decay_days)

            source_scores[source.source_id] = {
                "authority": auth,
                "citation": cit,
                "recency": rec,
            }
            authority_scores.append(auth)
            citation_scores.append(cit)
            recency_scores.append(rec)

        mean_auth = sum(authority_scores) / len(authority_scores)
        mean_cit = sum(citation_scores) / len(citation_scores)
        mean_rec = sum(recency_scores) / len(recency_scores)

        # Consistency score
        if self.check_consistency and len(contexts) >= 2:
            consistency = self._check_consistency_llm(question, contexts)
        else:
            # Heuristic: more sources that agree on authority → more consistent
            consistency = 1.0 - (max(authority_scores) - min(authority_scores))
            consistency = max(0.0, min(1.0, consistency))

        # Composite
        composite = (
            mean_auth * self.weights["authority"] +
            consistency * self.weights["consistency"] +
            mean_cit * self.weights["citation"] +
            mean_rec * self.weights["recency"]
        )

        result = SourceReliabilityResult(
            source_scores=source_scores,
            mean_authority=mean_auth,
            mean_citation=mean_cit,
            mean_recency=mean_rec,
            consistency_score=consistency,
            composite_score=composite,
            n_sources=len(sources),
            weights=self.weights,
        )

        logger.info(
            f"[source_reliability] composite={composite:.4f} | "
            f"authority={mean_auth:.4f} | citation={mean_cit:.4f} | "
            f"recency={mean_rec:.4f} | consistency={consistency:.4f}"
        )

        return MetricResult(
            metric_name=self.name,
            score=composite,
            raw=result.to_dict(),
            metadata={
                "n_sources": len(sources),
                "consistency_method": "llm" if self.check_consistency else "heuristic",
            },
        )

    def _check_consistency_llm(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        """
        LLM-based cross-source consistency check.
        Asks judge whether the sources agree with each other.
        """
        try:
            import anthropic
            client = anthropic.Anthropic()

            context_block = "\n\n".join(
                f"[Source {i+1}]: {ctx}" for i, ctx in enumerate(contexts[:5])
            )
            prompt = (
                f"Question: {question}\n\n"
                f"Retrieved sources:\n{context_block}\n\n"
                f"Do these sources agree with each other on the key facts relevant "
                f"to the question? Score from 0.0 (completely contradictory) to "
                f"1.0 (fully consistent). Reply with ONLY a float."
            )
            message = client.messages.create(
                model=self.judge_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            matches = re.findall(r"[0-9]*\.?[0-9]+", raw)
            if matches:
                return max(0.0, min(1.0, float(matches[0])))
        except Exception as e:
            logger.warning(f"[source_reliability] LLM consistency check failed: {e}")
        return 0.5

    async def score_async(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.score(question, contexts, answer, ground_truth, **kwargs),
        )
