"""
eval_engine/schemas.py

Evaluation data structures for the eval-engine SDK.

SOURCE: Consolidation RAG eval framework (your code), with the following changes:
  FIX-1: relevant_ids changed from set[str] to frozenset[str]
          Reason: set is not JSON-serializable; frozenset is consistent
          with frozen=True dataclass contract and prevents accidental mutation.
          Serialization helper added: RetrievalCase.relevant_ids_list property.
  FIX-3: GraphSnapshot gains three first-class fields before metadata dict:
          cycle_id, captured_at, experiment_id.
          Reason: untyped metadata dict cannot guarantee snapshot ordering
          for pre/post consolidation delta computation.

UNCHANGED from your original:
  - RetrievalResult (verbatim)
  - All docstrings and inline comments (verbatim)
  - All optional fields on GraphSnapshot (verbatim)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalCase:
    """Single benchmark query and its expected relevant IDs."""

    # Stable ID for the benchmark query.
    # This lets results and labels be matched reliably.
    query_id: str

    # The natural-language query being evaluated.
    query: str

    # FIX-1: Changed from set[str] to frozenset[str].
    # set is not JSON-serializable and is inconsistent with frozen=True.
    # frozenset is immutable and hashable — correct for a frozen dataclass.
    # Use .relevant_ids_list property when writing to JSONL.
    relevant_ids: frozenset[str]

    def __post_init__(self) -> None:
        # Coerce list or set inputs to frozenset at construction time.
        # Callers who pass a list or set get a frozenset transparently.
        if not isinstance(self.relevant_ids, frozenset):
            object.__setattr__(self, "relevant_ids", frozenset(self.relevant_ids))

    @property
    def relevant_ids_list(self) -> list[str]:
        """JSON-serializable form of relevant_ids. Use for JSONL output."""
        return sorted(self.relevant_ids)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalCase":
        """Construct from a raw dict (e.g. loaded from JSON dataset)."""
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            relevant_ids=frozenset(data.get("relevant_ids", [])),
        )


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved IDs for a benchmark query, ordered by rank."""

    # ID of the query this result belongs to.
    query_id: str

    # Retrieved item IDs in ranked order.
    # Index 0 is the top-ranked retrieval result.
    retrieved_ids: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """Construct from a raw dict."""
        return cls(
            query_id=data["query_id"],
            retrieved_ids=data.get("retrieved_ids", []),
        )


@dataclass(frozen=True)
class GraphSnapshot:
    """Small graph summary captured before or after refinement."""

    # Number of nodes in the graph at this point in time.
    node_count: int

    # Number of edges/relationships in the graph at this point in time.
    edge_count: int

    # Number of duplicate entities or relations detected.
    duplicate_count: int = 0

    # FIX-3: Three first-class fields added before metadata dict.
    # These are required for reliable pre/post consolidation delta matching.
    # Without cycle_id, two snapshots cannot be reliably ordered.

    # Dream cycle number. 0 = pre-consolidation baseline.
    # Increment by 1 per completed consolidation cycle.
    cycle_id: int | None = None

    # Unix timestamp (time.time()) when snapshot was captured.
    # Populated automatically by GraphSnapshot.capture() classmethod.
    captured_at: float | None = None

    # Ties this snapshot to an EvalConfig.experiment_id for traceability.
    experiment_id: str | None = None

    # Number of expected entities currently represented in the graph.
    covered_entities: int | None = None

    # Total number of entities expected for the benchmark or dataset.
    expected_entities: int | None = None

    # Number of valid relations currently represented in the graph.
    valid_relations: int | None = None

    # Total number of relations expected for the benchmark or dataset.
    expected_relations: int | None = None

    # Extra information: dataset version, hardware config, etc.
    # Does NOT store cycle_id, captured_at, experiment_id — those are
    # first-class fields above to enforce type safety.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        node_count: int,
        edge_count: int,
        cycle_id: int,
        experiment_id: str,
        duplicate_count: int = 0,
        covered_entities: int | None = None,
        expected_entities: int | None = None,
        valid_relations: int | None = None,
        expected_relations: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GraphSnapshot":
        """
        Preferred constructor for runtime snapshot capture.
        Automatically sets captured_at to current time.

        Usage:
            pre = GraphSnapshot.capture(
                node_count=1200, edge_count=4800,
                cycle_id=0, experiment_id="consolidation_rag_paper1_run1"
            )
            post = GraphSnapshot.capture(
                node_count=980, edge_count=3900,
                cycle_id=1, experiment_id="consolidation_rag_paper1_run1"
            )
        """
        return cls(
            node_count=node_count,
            edge_count=edge_count,
            duplicate_count=duplicate_count,
            cycle_id=cycle_id,
            captured_at=time.time(),
            experiment_id=experiment_id,
            covered_entities=covered_entities,
            expected_entities=expected_entities,
            valid_relations=valid_relations,
            expected_relations=expected_relations,
            metadata=metadata or {},
        )

    def is_before(self, other: "GraphSnapshot") -> bool:
        """
        Return True if this snapshot was captured before `other`.
        Uses cycle_id first (preferred), falls back to captured_at.
        Raises ValueError if neither field is set on either snapshot.
        """
        if self.cycle_id is not None and other.cycle_id is not None:
            return self.cycle_id < other.cycle_id
        if self.captured_at is not None and other.captured_at is not None:
            return self.captured_at < other.captured_at
        raise ValueError(
            "Cannot determine snapshot ordering: "
            "both cycle_id and captured_at are None. "
            "Use GraphSnapshot.capture() to set these fields automatically."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "duplicate_count": self.duplicate_count,
            "cycle_id": self.cycle_id,
            "captured_at": self.captured_at,
            "experiment_id": self.experiment_id,
            "covered_entities": self.covered_entities,
            "expected_entities": self.expected_entities,
            "valid_relations": self.valid_relations,
            "expected_relations": self.expected_relations,
            "metadata": self.metadata,
        }
