"""
eval_engine/metrics/source_reliability.py

Source Reliability Scoring — SCAFFOLD

STATUS: Scaffold. Interface locked. Implementation deferred to Tier 2.
        NOTE: Requires schema extension to schemas.py before implementation
        can proceed (source metadata fields needed on RetrievalResult).

RESEARCH QUESTION:
    How reliable are the sources that Consolidation RAG retrieves?
    Does post-consolidation retrieval surface more or less reliable
    sources than pre-consolidation baseline?

    This is distinct from faithfulness (answer grounded in context)
    and hallucination (claims unsupported by context):
        - Source reliability asks about the SOURCES themselves
        - Even a faithful, non-hallucinated answer can be unreliable
          if it's grounded in an unreliable source

PLANNED DIMENSIONS:
    1. Authority Score (0.0–1.0)
       Is the source from a recognized authoritative domain?
       (peer-reviewed, government, institutional vs. anonymous)

    2. Consistency Score (0.0–1.0)
       Does the source's claims align with other retrieved sources?
       Cross-source agreement as reliability proxy.

    3. Citation Quality (0.0–1.0)
       Does the source cite its own sources?
       Citation density as quality signal.

    4. Recency Score (0.0–1.0)
       How recent is the source?
       Decay function based on publication date.

SCHEMA DEPENDENCY:
    Requires source metadata on retrieved documents:
        - source_url or source_id
        - publication_date
        - domain / publisher
        - citation_count (optional)

    This means RetrievalResult in schemas.py needs extension OR
    a separate SourceMetadata dataclass alongside it.

    This is why source_reliability is Tier 2, not Tier 1 —
    the schema work is prerequisite.

PLANNED INTEGRATION:
    - Composite feeds into TrustScore as source_reliability component
    - Per-source breakdown in Longitudinal Eval Suite results appendix

PAPER PLACEMENT:
    Longitudinal Eval Suite or 3 — "Source Provenance and Reliability in Consolidated RAG"
    Strong fit for information quality / digital provenance research venues.

PRIOR WORK TO CITE:
    - MediaBias/FactCheck reliability taxonomy
    - NewsGuard credibility scoring
    - FEVER (Thorne et al., 2018) for fact verification methodology
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class SourceMetadata:
    """
    Metadata for a single retrieved source.
    SCAFFOLD — fields defined, not yet integrated into RetrievalResult.
    Extend schemas.py RetrievalResult to include List[SourceMetadata]
    before implementing this metric.
    """
    source_id: str
    source_url: str | None = None
    domain: str | None = None
    publication_date: str | None = None
    publisher: str | None = None
    citation_count: int | None = None
    is_peer_reviewed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceReliabilityMetric(BaseMetric):
    """
    Source Reliability Scoring.

    SCAFFOLD — returns NotImplemented result with research context.
    Implementation target: Tier 2.

    PREREQUISITE: Schema extension to RetrievalResult in schemas.py
    to include source metadata fields.
    """

    name = "source_reliability"

    def __init__(
        self,
        authority_weight: float = 0.35,
        consistency_weight: float = 0.30,
        citation_weight: float = 0.20,
        recency_weight: float = 0.15,
        recency_decay_days: int = 365,
    ) -> None:
        self.weights = {
            "authority": authority_weight,
            "consistency": consistency_weight,
            "citation": citation_weight,
            "recency": recency_weight,
        }
        self.recency_decay_days = recency_decay_days

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
        SCAFFOLD — not yet implemented.

        When implemented, scores reliability of retrieved sources
        across authority, consistency, citation quality, and recency.

        Requires source_metadata kwarg (list[SourceMetadata]).
        """
        logger.info(
            "[source_reliability] SCAFFOLD — implementation deferred to Tier 2. "
            "Prerequisite: schema extension to RetrievalResult."
        )
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            raw={},
            metadata={
                "status": "scaffold",
                "implementation_target": "Tier 2",
                "prerequisite": "Extend schemas.py RetrievalResult with SourceMetadata",
                "planned_dimensions": list(self.weights.keys()),
                "has_source_metadata": source_metadata is not None,
                "research_question": (
                    "How reliable are the sources Consolidation RAG retrieves? "
                    "Does post-consolidation retrieval surface more reliable sources?"
                ),
                "prior_work": [
                    "FEVER (Thorne et al., 2018)",
                    "MediaBias/FactCheck reliability taxonomy",
                    "NewsGuard credibility scoring",
                ],
            },
        )
