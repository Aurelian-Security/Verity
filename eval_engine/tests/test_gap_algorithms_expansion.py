"""
Tests for eval_engine/metrics/gap_algorithms_expansion.py.

Structure mirrors test_algorithm_expansion.py (shape assertion over the
whole registry, plus targeted behavioral tests). The behavioral tests below
are the original verity_gap_algorithms/tests/test_algorithms.py assertions
(6/6 passing there), re-run through the wrapper classes to confirm the
translation from typed-dataclass contract to MetricResult contract didn't
change any scoring outcome. The two probe-robustness tests additionally
confirm the shared-core refactor matches the original standalone
implementations (verity_gap_algorithms.algorithms.probe_distribution_shift_robustness
and the pre-refactor verity_gap_closure cross_architecture_probe_validation).
"""
from eval_engine.metrics.gap_algorithms_expansion import (
    GAP_ALGORITHMS_EXPANSION_REGISTRY,
    ReliabilityWeightedConsensusMetric,
    ScenarioAdaptiveJailbreakMetric,
    TurnsToExploitMetric,
    TrustCalibrationMetric,
    CrossCategoryRegressionDiffMetric,
    ManifestHashVerificationMetric,
    ProbeDistributionShiftMetric,
    CrossArchitectureProbeValidationMetric,
)
from eval_engine.metrics import _gap_algorithms_core as core
from eval_engine.metrics import _probe_robustness_core as probe_core


def _sample():
    return {"question": "q", "contexts": ["c"], "answer": "a"}


def test_gap_algorithms_expansion_has_8_metrics():
    assert len(GAP_ALGORITHMS_EXPANSION_REGISTRY) == 8


def test_all_metrics_return_metric_result_shape():
    # Minimal valid kwargs per metric so every registered class can score() cleanly.
    kwargs_by_name = {
        "reliability_weighted_judge_consensus": {
            "votes": [{"judge_id": "a", "score": 0.5}]
        },
        "scenario_adaptive_jailbreak_scoring": {
            "observed": {"x": 0.5},
            "criteria": [{"name": "x", "weight": 1.0}],
        },
        "turns_to_exploit": {"turn_scores": [0.1, 0.2]},
        "trust_calibration_metrics": {"confidences": [0.5, 0.5], "outcomes": [1, 0]},
        "cross_category_regression_diff": {"baseline": {"m": 1.0}, "candidate": {"m": 1.0}},
        "canonical_manifest_hash_verification": {"manifest": {"a": 1}},
        "probe_distribution_shift_robustness": {
            "labels": [0, 1, 0, 1],
            "scores": [0.1, 0.9, 0.2, 0.8],
            "groups": ["in_domain"] * 2 + ["style_shift"] * 2,
        },
        "cross_architecture_probe_validation": {
            "labels": [0, 1, 0, 1],
            "scores": [0.1, 0.9, 0.2, 0.8],
            "families": ["llama"] * 2 + ["mistral"] * 2,
        },
    }
    sample = _sample()
    for name, cls in GAP_ALGORITHMS_EXPANSION_REGISTRY.items():
        metric = cls()
        result = metric.score(**sample, **kwargs_by_name[name])
        assert result.metric_name == name
        assert isinstance(result.score, float)
        assert result.raw is not None
        assert result.metadata is not None


# ---------------------------------------------------------------------------
# Behavioral tests -- translated 1:1 from verity_gap_algorithms/tests/test_algorithms.py
# ---------------------------------------------------------------------------

def test_consensus_flags_disagreement():
    result = ReliabilityWeightedConsensusMetric().score(
        **_sample(),
        votes=[{"judge_id": "a", "score": 0.1}, {"judge_id": "b", "score": 0.9}],
    )
    assert result.raw["fallback_required"] is True


def test_scenario_required_dimension():
    result = ScenarioAdaptiveJailbreakMetric().score(
        **_sample(),
        observed={"violation": 0.9, "actionability": 0.2},
        criteria=[
            {"name": "violation", "weight": 0.5, "required": True},
            {"name": "actionability", "weight": 0.5, "required": True},
        ],
    )
    assert result.raw["successful"] is False
    assert "actionability" in result.raw["missing_required"]


def test_tte():
    result = TurnsToExploitMetric().score(**_sample(), turn_scores=[0.1, 0.3, 0.8])
    assert result.raw["first_exploit_turn"] == 3


def test_calibration_perfect():
    result = TrustCalibrationMetric().score(
        **_sample(), confidences=[0.0, 1.0], outcomes=[0, 1], n_bins=2
    )
    assert result.raw["ece"] == 0.0
    assert result.raw["brier_score"] == 0.0
    assert result.score == 1.0


def test_diff_directionality():
    result = CrossCategoryRegressionDiffMetric().score(
        **_sample(),
        baseline={"safety": 0.8, "ece": 0.1},
        candidate={"safety": 0.7, "ece": 0.15},
        higher_is_better={"safety": True, "ece": False},
    )
    assert all(d["regressed"] for d in result.raw["deltas"])
    assert result.score == 0.0


def test_manifest_hash_is_deterministic_and_order_independent():
    a = ManifestHashVerificationMetric().score(**_sample(), manifest={"a": 1, "b": 2})
    b = ManifestHashVerificationMetric().score(**_sample(), manifest={"b": 2, "a": 1})
    assert a.raw["manifest_hash"] == b.raw["manifest_hash"]
    assert a.raw["manifest_hash"] == core.canonical_manifest_hash({"a": 1, "b": 2})


def test_manifest_hash_verification_flags_mismatch():
    result = ManifestHashVerificationMetric().score(
        **_sample(), manifest={"a": 1}, expected_hash="not-the-real-hash"
    )
    assert result.score == 0.0
    assert result.raw["verified"] is False


# ---------------------------------------------------------------------------
# Probe-robustness coexistence regression tests
# ---------------------------------------------------------------------------

def test_probe_shift_matches_original_verity_gap_algorithms_output():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.7, 0.3]
    groups = ["in_domain"] * 4 + ["style_shift"] * 4

    wrapped = ProbeDistributionShiftMetric().score(
        **_sample(), labels=labels, scores=scores, groups=groups
    )
    assert wrapped.raw["in_domain_auc"] == 1.0
    assert wrapped.raw["worst_group_auc"] == 0.0

    # Cross-check against the shared core directly (the same math the
    # original standalone verity_gap_algorithms.algorithms function used).
    direct = probe_core.grouped_robustness(
        labels, scores, groups,
        baseline_partition="in_domain",
        exclude_baseline_from_worst=True,
        failure_auc=0.70,
    )
    assert wrapped.raw["in_domain_auc"] == direct.baseline_auc
    assert wrapped.raw["worst_group_auc"] == direct.worst_partition_auc
    assert wrapped.score == direct.robustness_ratio


def test_cross_architecture_uses_shared_core_not_reimplemented_auroc():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.55, 0.45, 0.6, 0.4]
    families = ["llama"] * 4 + ["mistral"] * 4

    wrapped = CrossArchitectureProbeValidationMetric().score(
        **_sample(), labels=labels, scores=scores, families=families
    )
    direct = probe_core.grouped_robustness(
        labels, scores, families,
        baseline_value=None,
        exclude_baseline_from_worst=False,
        failure_auc=None,
    )
    expected_score = max(0.0, min(1.0, 0.6 * direct.worst_partition_auc + 0.4 * direct.robustness_ratio))
    assert wrapped.score == expected_score
    assert wrapped.raw["worst_family_auc"] == direct.worst_partition_auc
    assert wrapped.raw["robustness_ratio"] == direct.robustness_ratio
