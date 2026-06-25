"""
eval_engine/tests/test_scaffolds.py

Scaffold smoke tests — Tier 2 + Tier 3.

Verifies that all scaffold metrics:
    1. Are importable and instantiable
    2. Are registered in MetricsRegistry
    3. Return a MetricResult (not raise) when score() is called
    4. Include scaffold status metadata
    5. Document their research question and prior work

These tests ensure scaffold interfaces remain stable as implementation
proceeds, and that no scaffold silently breaks the registry.
"""

from __future__ import annotations

import pytest

SCAFFOLD_METRICS = [
    # Tier 2
    "consistency",
    "constitutional_eval",
    "model_written_eval",
    "source_reliability",
    # Tier 3
    "goal_misgeneralization",
    "deceptive_alignment",
]

SAMPLE_INPUTS = dict(
    question="What does the consolidation phase do in a memory-augmented RAG system?",
    contexts=["The consolidation phase prunes low-weight edges in the knowledge graph."],
    answer="The consolidation phase removes redundant graph connections.",
)


class TestScaffoldRegistry:

    def test_all_scaffolds_in_registry(self):
        """All scaffold metrics registered in MetricsRegistry."""
        from eval_engine.metrics import registry
        for name in SCAFFOLD_METRICS:
            assert name in registry.available, f"'{name}' missing from registry"

    def test_registry_total_count(self):
        """Registry has expected total metric count (implemented + scaffolds)."""
        from eval_engine.metrics import registry
        # 17 implemented + 6 scaffolds = 23
        assert len(registry.available) >= 23

    def test_scaffold_instantiation(self):
        """All scaffolds instantiate without error."""
        from eval_engine.metrics import registry
        for name in SCAFFOLD_METRICS:
            m = registry.get(name)
            assert m is not None
            assert m.name == name


class TestScaffoldBehavior:

    @pytest.mark.parametrize("metric_name", SCAFFOLD_METRICS)
    def test_scaffold_returns_metric_result(self, metric_name):
        """Each scaffold returns MetricResult without raising."""
        from eval_engine.metrics import registry
        from eval_engine.metrics.base import MetricResult
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert isinstance(result, MetricResult)
        assert result.metric_name == metric_name

    @pytest.mark.parametrize("metric_name", SCAFFOLD_METRICS)
    def test_scaffold_has_status_metadata(self, metric_name):
        """Each scaffold result contains status='scaffold' in metadata."""
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert result.metadata.get("status") == "scaffold", (
            f"'{metric_name}' missing status='scaffold' in metadata"
        )

    @pytest.mark.parametrize("metric_name", SCAFFOLD_METRICS)
    def test_scaffold_has_research_question(self, metric_name):
        """Each scaffold documents its research question."""
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert "research_question" in result.metadata, (
            f"'{metric_name}' missing research_question in metadata"
        )
        assert len(result.metadata["research_question"]) > 20

    @pytest.mark.parametrize("metric_name", SCAFFOLD_METRICS)
    def test_scaffold_has_implementation_target(self, metric_name):
        """Each scaffold declares its implementation tier."""
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert "implementation_target" in result.metadata, (
            f"'{metric_name}' missing implementation_target in metadata"
        )
        assert result.metadata["implementation_target"] in ("Tier 2", "Tier 3")

    @pytest.mark.parametrize("metric_name", SCAFFOLD_METRICS)
    def test_scaffold_has_prior_work(self, metric_name):
        """Each scaffold cites prior work."""
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert "prior_work" in result.metadata, (
            f"'{metric_name}' missing prior_work citations in metadata"
        )
        assert len(result.metadata["prior_work"]) > 0

    @pytest.mark.parametrize("metric_name", SCAFFOLD_METRICS)
    def test_scaffold_score_is_zero(self, metric_name):
        """All scaffolds return score=0.0 (placeholder)."""
        from eval_engine.metrics import registry
        m = registry.get(metric_name)
        result = m.score(**SAMPLE_INPUTS)
        assert result.score == 0.0, (
            f"'{metric_name}' should return 0.0 placeholder score"
        )


class TestTier2Scaffolds:

    def test_consistency_n_runs_config(self):
        """ConsistencyMetric accepts n_runs parameter."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric(n_runs=10)
        assert m.n_runs == 10

    def test_consistency_raises_on_multiple_runs(self):
        """score_multiple_runs() raises NotImplementedError."""
        from eval_engine.metrics.consistency import ConsistencyMetric
        m = ConsistencyMetric()
        with pytest.raises(NotImplementedError):
            m.score_multiple_runs("Q", None, None)

    def test_constitutional_dimensions_configurable(self):
        """ConstitutionalEvalMetric accepts custom dimensions."""
        from eval_engine.metrics.constitutional_eval import ConstitutionalEvalMetric
        m = ConstitutionalEvalMetric(dimensions=["harmlessness", "honesty"])
        assert m.dimensions == ["harmlessness", "honesty"]

    def test_constitutional_default_dimensions(self):
        """Default dimensions include all five planned dimensions."""
        from eval_engine.metrics.constitutional_eval import (
            ConstitutionalEvalMetric, CONSTITUTIONAL_DIMENSIONS
        )
        m = ConstitutionalEvalMetric()
        assert set(m.dimensions) == set(CONSTITUTIONAL_DIMENSIONS)

    def test_mwe_default_governance_rubric(self):
        """ModelWrittenEvalMetric.default_governance_rubric() returns configured instance."""
        from eval_engine.metrics.model_written_eval import ModelWrittenEvalMetric
        m = ModelWrittenEvalMetric.default_governance_rubric()
        assert len(m.rubric) == 4
        required = [c for c in m.rubric if c.required]
        assert len(required) == 1
        assert required[0].name == "scope_adherence"

    def test_source_reliability_weights_sum(self):
        """SourceReliabilityMetric weights sum to 1.0."""
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        m = SourceReliabilityMetric()
        assert abs(sum(m.weights.values()) - 1.0) < 0.001

    def test_source_reliability_prerequisite_documented(self):
        """source_reliability metadata documents schema prerequisite."""
        from eval_engine.metrics.source_reliability import SourceReliabilityMetric
        m = SourceReliabilityMetric()
        result = m.score(**SAMPLE_INPUTS)
        assert "prerequisite" in result.metadata
        assert "schemas.py" in result.metadata["prerequisite"]


class TestTier3Scaffolds:

    def test_goal_misgeneralization_latentids_not_mentioned_yet(self):
        """GoalMisgeneralizationMetric documents its prerequisites."""
        from eval_engine.metrics.goal_misgeneralization import GoalMisgeneralizationMetric
        m = GoalMisgeneralizationMetric()
        result = m.score(**SAMPLE_INPUTS)
        assert "prerequisites" in result.metadata
        assert len(result.metadata["prerequisites"]) >= 3

    def test_deceptive_alignment_latentids_connection(self):
        """DeceptiveAlignmentMetric documents LatentIDS connection."""
        from eval_engine.metrics.deceptive_alignment import DeceptiveAlignmentMetric
        m = DeceptiveAlignmentMetric()
        result = m.score(**SAMPLE_INPUTS)
        assert "latentids_connection" in result.metadata

    def test_deceptive_alignment_dimensions(self):
        """DeceptiveAlignmentMetric has all four planned dimensions."""
        from eval_engine.metrics.deceptive_alignment import (
            DeceptiveAlignmentMetric, DECEPTIVE_ALIGNMENT_DIMENSIONS
        )
        m = DeceptiveAlignmentMetric()
        assert set(m.dimensions) == set(DECEPTIVE_ALIGNMENT_DIMENSIONS)

    def test_tier3_target_venues_documented(self):
        """Tier 3 scaffolds document target publication venues."""
        from eval_engine.metrics import registry
        for name in ["goal_misgeneralization", "deceptive_alignment"]:
            m = registry.get(name)
            result = m.score(**SAMPLE_INPUTS)
            assert "target_venues" in result.metadata
            assert len(result.metadata["target_venues"]) > 0
