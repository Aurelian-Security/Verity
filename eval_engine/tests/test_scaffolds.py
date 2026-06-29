"""
eval_engine/tests/test_scaffolds.py

Scaffold and Tier 2 implementation tests.

Tier 2 metrics (consistency, constitutional_eval, model_written_eval,
source_reliability) are now IMPLEMENTED. Tests updated accordingly.

Tier 3 metrics (goal_misgeneralization, deceptive_alignment) remain
scaffolded — tested for scaffold contract compliance.
"""

from __future__ import annotations

import pytest

# Tier 3 only — still scaffolded
TIER3_SCAFFOLD_METRICS = [
    "goal_misgeneralization",
    "deceptive_alignment",
]

# Tier 2 — now implemented
TIER2_IMPLEMENTED_METRICS = [
    "consistency",
    "constitutional_eval",
    "model_written_eval",
    "source_reliability",
]

SAMPLE_INPUTS = dict(
    question="What does the consolidation phase do in a memory-augmented RAG system?",
    contexts=["The consolidation phase prunes low-weight edges in the knowledge graph."],
    answer="The consolidation phase removes redundant graph connections.",
)


# =============================================================================
# Registry — all metrics present
# =============================================================================

class TestRegistry:

    def test_all_tier2_in_registry(self):
        from eval_engine.metrics import registry
        for name in TIER2_IMPLEMENTED_METRICS:
            assert name in registry.available, f"'{name}' missing from registry"

    def test_all_tier3_in_registry(self):
        from eval_engine.metrics import registry
        for name in TIER3_SCAFFOLD_METRICS:
            assert name in registry.available, f"'{name}' missing from registry"

    def test_registry_total_count(self):
        from eval_engine.metrics import registry
        assert len(registry.available) >= 23

    def test_all_metrics_instantiate(self):
        from eval_engine.metrics import registry
        for name in TIER2_IMPLEMENTED_METRICS + TIER3_SCAFFOLD_METRICS:
            m = registry.get(name)
            assert m is not None
            assert m.name == name


# =============================================================================
# Tier 3 scaffolds — unchanged scaffold contract
# =============================================================================

class TestTier3Scaffolds:

    @pytest.mark.parametrize("metric_name", TIER3_SCAFFOLD_METRICS)
    def test_scaffold_returns_metric_result(self, metric_name):
        from eval_engine.metrics import registry
        from eval_engine.metrics.base import MetricResult
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert isinstance(result, MetricResult)
        assert result.metric_name == metric_name

    @pytest.mark.parametrize("metric_name", TIER3_SCAFFOLD_METRICS)
    def test_scaffold_has_status_metadata(self, metric_name):
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert result.metadata.get("status") == "scaffold"

    @pytest.mark.parametrize("metric_name", TIER3_SCAFFOLD_METRICS)
    def test_scaffold_has_research_question(self, metric_name):
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert "research_question" in result.metadata

    @pytest.mark.parametrize("metric_name", TIER3_SCAFFOLD_METRICS)
    def test_scaffold_has_prior_work(self, metric_name):
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert "prior_work" in result.metadata
        assert len(result.metadata["prior_work"]) > 0

    @pytest.mark.parametrize("metric_name", TIER3_SCAFFOLD_METRICS)
    def test_scaffold_score_is_zero(self, metric_name):
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert result.score == 0.0

    def test_goal_misgeneralization_prerequisites(self):
        from eval_engine.metrics.goal_misgeneralization import GoalMisgeneralizationMetric
        m = GoalMisgeneralizationMetric()
        result = m.score(**SAMPLE_INPUTS)
        assert "prerequisites" in result.metadata

    def test_deceptive_alignment_latentids_connection(self):
        from eval_engine.metrics.deceptive_alignment import DeceptiveAlignmentMetric
        m = DeceptiveAlignmentMetric()
        result = m.score(**SAMPLE_INPUTS)
        assert "latentids_connection" in result.metadata

    def test_tier3_target_venues_documented(self):
        from eval_engine.metrics import registry
        for name in TIER3_SCAFFOLD_METRICS:
            m = registry.get(name)
            result = m.score(**SAMPLE_INPUTS)
            assert "target_venues" in result.metadata


# =============================================================================
# Tier 2 — now implemented, test real behavior
# =============================================================================

class TestConsistencyImplemented:

    def test_consistency_instantiates(self):
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(n_runs=5, mode="jaccard")
        assert m.name == "consistency"
        assert m.n_runs == 5

    def test_consistency_score_multiple_runs_jaccard(self):
        """score_multiple_runs() returns real score in [0,1] using jaccard mode."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        answers = [
            "The consolidation phase improves retrieval precision.",
            "Consolidation improves retrieval by pruning edges.",
            "The consolidation cycle enhances retrieval quality.",
        ]
        result = m.score_multiple_runs(question="Q", answers=answers)
        assert 0.0 <= result.score <= 1.0
        assert result.raw["n_runs"] == 3
        assert result.raw["mode"] == "jaccard"

    def test_consistency_identical_answers(self):
        """Identical answers → score = 1.0."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        answer = "The consolidation phase improves retrieval."
        result = m.score_multiple_runs("Q", [answer, answer, answer])
        assert abs(result.score - 1.0) < 0.01

    def test_consistency_drift_detected(self):
        """Very different answers → score < 1.0."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        answers = [
            "The consolidation phase improves retrieval quality.",
            "Bananas are a tropical fruit grown in warm climates.",
            "The French Revolution began in 1789.",
        ]
        result = m.score_multiple_runs("Q", answers)
        assert result.score < 0.8

    def test_consistency_single_answer(self):
        """Single answer → trivially consistent (1.0)."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        result = m.score_multiple_runs("Q", ["Single answer."])
        assert result.score == 1.0

    def test_consistency_accumulator_mode(self):
        """Accumulate answers via score(), then compute_consistency_report()."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        for i in range(3):
            m.score(question="Q", contexts=["C"], answer=f"Answer variation {i}")
        assert len(m._answer_buffer) == 3
        report = m.compute_consistency_report()
        assert 0.0 <= report.score <= 1.0
        assert len(m._answer_buffer) == 0   # cleared after report

    def test_consistency_csi_computed(self):
        """Consolidation Stability Index computed when pre_answers provided."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        post_answers = [
            "The consolidation phase improves retrieval precision.",
            "Consolidation improves retrieval by pruning redundant edges.",
            "The consolidation cycle enhances retrieval quality.",
        ]
        pre_answers = [
            "Retrieval before consolidation has lower precision overall.",
            "Before consolidation the retrieval quality is degraded.",
            "Pre-consolidation retrieval shows lower quality metrics.",
        ]
        result = m.score_multiple_runs("Q", post_answers, pre_consolidation_answers=pre_answers)
        # CSI may be None if pre_drift is 0 — just verify the field exists in raw
        assert "consolidation_stability_index" in result.raw

    def test_consistency_score_single_not_implemented(self):
        """score() in accumulation mode returns placeholder until report called."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        result = m.score("Q", ["C"], "Answer")
        assert result.score == 0.0
        assert "accumulated_runs" in result.metadata

    def test_consistency_no_longer_raises_on_multiple_runs(self):
        """score_multiple_runs() no longer raises NotImplementedError."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(mode="jaccard")
        result = m.score_multiple_runs("Q", ["A1", "A2"])
        assert result.metric_name == "consistency"


class TestConstitutionalEvalImplemented:

    def test_constitutional_eval_instantiates(self):
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dry_run=True)
        assert m.name == "constitutional_eval"

    def test_constitutional_eval_dry_run(self):
        """Dry-run returns real scores without API calls."""
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert 0.0 <= result.score <= 1.0
        assert result.raw["dry_run"] is True
        assert "dimension_scores" in result.raw

    def test_constitutional_eval_all_dimensions_scored(self):
        """All 5 dimensions present in dry-run result."""
        from eval_engine.metrics.constitutional_eval import (
            ConstitutionalEvalMetric, CONSTITUTIONAL_DIMENSIONS
        )
        m = ConstitutionalEvalMetric(dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        for dim in CONSTITUTIONAL_DIMENSIONS:
            assert dim in result.raw["dimension_scores"]

    def test_constitutional_eval_custom_dimensions(self):
        """Custom subset of dimensions works."""
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dimensions=["harmlessness", "honesty"], dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert set(result.raw["dimension_scores"].keys()) == {"harmlessness", "honesty"}

    def test_constitutional_eval_weights_sum_to_one(self):
        """Weights normalize correctly."""
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dry_run=True)
        assert abs(sum(m.weights.values()) - 1.0) < 0.001

    def test_constitutional_eval_score_in_range(self):
        """Composite score always in [0, 1]."""
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert 0.0 <= result.score <= 1.0

    def test_constitutional_eval_not_scaffold(self):
        """Dry-run result does not have scaffold status."""
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert result.metadata.get("status") != "scaffold"


class TestModelWrittenEvalImplemented:

    def test_mwe_instantiates(self):
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.default_governance_rubric(dry_run=True)
        assert m.name == "model_written_eval"

    def test_mwe_dry_run(self):
        """Dry-run returns real score without API calls."""
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.default_governance_rubric(dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert 0.0 <= result.score <= 1.0
        assert result.raw["dry_run"] is True

    def test_mwe_governance_rubric_criteria(self):
        """Default governance rubric has 4 criteria."""
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.default_governance_rubric(dry_run=True)
        assert len(m.rubric) == 4

    def test_mwe_research_rubric_exists(self):
        """Research paper rubric classmethod available."""
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.research_paper_rubric(dry_run=True)
        assert len(m.rubric) == 4

    def test_mwe_required_criterion_in_governance(self):
        """scope_adherence is required in governance rubric."""
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.default_governance_rubric(dry_run=True)
        required = [c for c in m.rubric if c.required]
        assert len(required) == 1
        assert required[0].name == "scope_adherence"

    def test_mwe_custom_rubric(self):
        """Custom rubric with single criterion works."""
        from eval_engine.metrics.model_written_eval import (
            ModelWrittenEvalMetric, RubricCriterion
        )
        m = ModelWrittenEvalMetric(
            rubric=[RubricCriterion(name="clarity", prompt="Is this clear?", type="binary")],
            dry_run=True,
        )
        result = m.score(**SAMPLE_INPUTS)
        assert "clarity" in result.raw["criterion_scores"]

    def test_mwe_no_rubric_returns_error(self):
        """Empty rubric returns error result."""
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric(rubric=[], dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert not result.succeeded

    def test_mwe_score_in_range(self):
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.default_governance_rubric(dry_run=True)
        result = m.score(**SAMPLE_INPUTS)
        assert 0.0 <= result.score <= 1.0


class TestSourceReliabilityImplemented:

    def test_source_reliability_instantiates(self):
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        m = SourceReliabilityMetric()
        assert m.name == "source_reliability"

    def test_source_reliability_with_metadata(self):
        """Score with SourceMetadata returns real composite score."""
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        from eval_engine.schemas import SourceMetadata
        m = SourceReliabilityMetric()
        sources = [
            SourceMetadata(
                source_id="doc_1",
                domain="arxiv.org",
                publication_date="2024-01-15",
                citation_count=150,
                is_peer_reviewed=True,
            ),
            SourceMetadata(
                source_id="doc_2",
                domain="wikipedia.org",
                publication_date="2022-06-01",
                citation_count=0,
                is_peer_reviewed=False,
            ),
        ]
        result = m.score(**SAMPLE_INPUTS, source_metadata=sources)
        assert 0.0 <= result.score <= 1.0
        assert result.raw["n_sources"] == 2

    def test_source_reliability_no_metadata_warning(self):
        """Missing metadata returns 0.0 score with warning metadata."""
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        m = SourceReliabilityMetric()
        result = m.score(**SAMPLE_INPUTS)
        assert result.score == 0.0
        assert result.metadata.get("has_source_metadata") is False

    def test_source_reliability_from_result(self):
        """score_from_result() works with RetrievalResult."""
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        from eval_engine.schemas import RetrievalResult, SourceMetadata
        m = SourceReliabilityMetric()
        sources = [
            SourceMetadata(source_id="doc_1", domain="nature.com",
                           publication_date="2024-03-01", citation_count=500,
                           is_peer_reviewed=True),
        ]
        rr = RetrievalResult(query_id="q1", retrieved_ids=["doc_1"], source_metadata=sources)
        result = m.score_from_result(rr, question="Q", contexts=["C"])
        assert 0.0 <= result.score <= 1.0

    def test_source_reliability_peer_reviewed_higher(self):
        """Peer-reviewed source scores higher than non-peer-reviewed."""
        from eval_engine.metrics.source_reliability import _authority_score
        from eval_engine.schemas import SourceMetadata
        peer = SourceMetadata(source_id="a", domain="nature.com", is_peer_reviewed=True)
        non_peer = SourceMetadata(source_id="b", domain="nature.com", is_peer_reviewed=False)
        assert _authority_score(peer) > _authority_score(non_peer)

    def test_citation_score_zero_for_none(self):
        from eval_engine.metrics.source_reliability import _citation_score
        assert _citation_score(None) == 0.0
        assert _citation_score(0) == 0.0

    def test_citation_score_increases_with_count(self):
        from eval_engine.metrics.source_reliability import _citation_score
        assert _citation_score(10) < _citation_score(100) < _citation_score(1000)

    def test_recency_score_recent_higher(self):
        from eval_engine.metrics.source_reliability import _recency_score
        recent = _recency_score("2025-01-01")
        old = _recency_score("2010-01-01")
        assert recent > old

    def test_recency_score_unknown_date(self):
        from eval_engine.metrics.source_reliability import _recency_score
        assert _recency_score(None) == 0.5

    def test_source_reliability_weights_sum(self):
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        m = SourceReliabilityMetric()
        assert abs(sum(m.weights.values()) - 1.0) < 0.001

    def test_schema_source_metadata_in_schemas(self):
        """SourceMetadata importable from schemas.py."""
        from eval_engine.schemas import SourceMetadata
        s = SourceMetadata(source_id="test", domain="arxiv.org")
        assert s.source_id == "test"

    def test_retrieval_result_accepts_source_metadata(self):
        """RetrievalResult.source_metadata field works."""
        from eval_engine.schemas import RetrievalResult, SourceMetadata
        sources = [SourceMetadata(source_id="doc_1")]
        rr = RetrievalResult(query_id="q1", retrieved_ids=["doc_1"], source_metadata=sources)
        assert rr.source_metadata is not None
        assert len(rr.source_metadata) == 1
