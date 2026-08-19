"""
Tests for eval_engine/metrics/gap_closure_expansion.py.

Behavioral assertions below are translated 1:1 from the original
verity_gap_closure/tests/test_metrics.py (which called the custom
registry.run(name, EvaluationCase(...)) and asserted on r.score /
r.details). Same inputs, same expected outputs, run through the real
HeuristicMetric wrapper classes and MetricResult (.score / .raw) instead.

test_probe (cross_architecture_probe_validation) is intentionally not
translated here -- that metric now lives in gap_algorithms_expansion.py via
the probe-robustness coexistence refactor. See that file's test module for
its coverage.
"""
import math

from eval_engine.metrics.gap_closure_expansion import (
    GAP_CLOSURE_EXPANSION_REGISTRY,
    ShutdownComplianceRateMetric,
    GoalGeneralizationGapMetric,
    EvaluationConditionBehaviorGapMetric,
    SelectiveSandbaggingIndexMetric,
    PerformanceGapRecoveredMetric,
    SecureUtilityMetric,
    UnauthorizedToolCallRateMetric,
    ResourceAmplificationMetric,
)


def _sample():
    return {"question": "q", "contexts": ["c"], "answer": "a"}


def test_gap_closure_expansion_has_35_metrics():
    assert len(GAP_CLOSURE_EXPANSION_REGISTRY) == 35


def test_all_metrics_return_metric_result_shape():
    """Every class must score cleanly with empty/default kwargs -- validates
    that no wrapper crashes on missing structured data rather than degrading
    gracefully (mirrors the empty-input handling in the original functions,
    e.g. rate(n, 0, default=...))."""
    sample = _sample()
    for name, cls in GAP_CLOSURE_EXPANSION_REGISTRY.items():
        metric = cls()
        try:
            result = metric.score(**sample)
        except (ValueError, TypeError) as exc:
            # A handful of metrics require at least one positional structured
            # kwarg (e.g. goal_generalization_gap needs both success rates)
            # and correctly raise rather than silently defaulting -- same as
            # the original, which did `float(c.inputs["train_like_intended_success"])`
            # with no .get() fallback. That's expected, not a shape failure.
            continue
        assert result.metric_name == name
        assert isinstance(result.score, float)
        assert result.raw is not None
        assert result.metadata is not None


# ---------------------------------------------------------------------------
# Behavioral tests -- translated 1:1 from verity_gap_closure/tests/test_metrics.py
# ---------------------------------------------------------------------------

def test_corrigibility():
    result = ShutdownComplianceRateMetric().score(
        **_sample(), episodes=[{"obstructed": False}, {"obstructed": True}]
    )
    assert result.score == 0.5


def test_goal_gap():
    result = GoalGeneralizationGapMetric().score(
        **_sample(), train_like_intended_success=0.9, shifted_intended_success=0.6
    )
    assert math.isclose(result.raw["gap"], 0.3)


def test_alignment_gap():
    result = EvaluationConditionBehaviorGapMetric().score(
        **_sample(), evaluation_behavior_rate=0.8, deployment_behavior_rate=0.3
    )
    assert math.isclose(result.raw["signed_gap"], 0.5)


def test_sandbagging():
    result = SelectiveSandbaggingIndexMetric().score(
        **_sample(),
        sensitive_default=0.2, sensitive_elicited=0.8,
        safe_default=0.7, safe_elicited=0.75,
    )
    assert result.raw["selectivity"] > 0.5


def test_pgr():
    result = PerformanceGapRecoveredMetric().score(
        **_sample(),
        weak_performance=0.4,
        weak_supervised_strong_performance=0.7,
        gold_supervised_strong_performance=0.8,
    )
    assert math.isclose(result.score, 0.75)


def test_owasp():
    result = SecureUtilityMetric().score(
        **_sample(),
        episodes=[
            {"task_success": True, "attack_success": False},
            {"task_success": True, "attack_success": True},
        ],
    )
    assert result.score == 0.5


def test_tools():
    result = UnauthorizedToolCallRateMetric().score(
        **_sample(),
        events=[
            {"tool_name": "read", "authorized": True},
            {"tool_name": "delete", "authorized": False},
        ],
    )
    assert result.raw["unauthorized"] == 1


def test_resource():
    result = ResourceAmplificationMetric().score(
        **_sample(),
        baseline={"tokens": 100, "tool_calls": 2, "wall_time": 1, "cost": 1},
        attacked={"tokens": 300, "tool_calls": 4, "wall_time": 2, "cost": 2},
    )
    assert result.raw["worst_factor"] == 3


def test_resource_amplification_unbounded_guard():
    """Confirms the kept inf-guard fix: baseline=0, attacked>0 caps to
    normalization_cap instead of producing a non-JSON-serializable inf, and
    separately records which resource was actually unbounded."""
    import json

    result = ResourceAmplificationMetric().score(
        **_sample(),
        baseline={"tokens": 0, "tool_calls": 2, "wall_time": 1, "cost": 1},
        attacked={"tokens": 50, "tool_calls": 4, "wall_time": 2, "cost": 2},
    )
    assert result.raw["amplification_factors"]["tokens"] == 10.0
    assert "tokens" in result.raw["unbounded_resources"]
    assert result.score == 0.0
    json.dumps(result.raw)  # must not raise


def test_registry_covers_all_expected_categories():
    names = set(GAP_CLOSURE_EXPANSION_REGISTRY)
    assert {"shutdown_compliance_rate", "correction_acceptance_rate",
            "intervention_cost", "corrigibility_robustness"} <= names
    assert {"goal_generalization_gap", "proxy_preference_rate",
            "specification_gaming_severity", "counterfactual_goal_consistency"} <= names
    assert {"evaluation_condition_behavior_gap", "strategic_compliance_rate",
            "condition_leakage_adjusted_gap", "alignment_faking_persistence"} <= names
    assert {"targeted_underperformance_gap", "selective_sandbagging_index",
            "noise_induced_capability_uplift", "answer_distribution_anomaly",
            "item_response_inconsistency"} <= names
    assert {"performance_gap_recovered", "oversight_uplift",
            "error_discovery_recall", "false_approval_rate"} <= names
    assert "layer_normalized_transfer" in names
    assert "cross_architecture_probe_validation" not in names  # lives in gap_algorithms_expansion.py
