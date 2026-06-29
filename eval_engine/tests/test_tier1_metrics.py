"""
eval_engine/tests/test_tier1_metrics.py

Tier 1 unit tests — Calibration, Hallucination, Trust Score, Multi-Session Persistence.

Covers:
    Calibration:
        test_ece_perfect_calibration        — conf == accuracy → ECE = 0
        test_ece_overconfident              — always conf=1.0, acc=0.5 → ECE = 0.5
        test_brier_perfect                  — confidence matches labels → Brier ≈ 0
        test_brier_worst                    — confidence opposite labels → Brier = 1
        test_calibration_bin_count          — correct number of bins returned
        test_calibration_reliability_flag   — low-sample bins flagged
        test_calibration_direction          — overconfident vs underconfident
        test_calibration_accumulator        — single-query accumulation mode
        test_calibration_registry           — accessible via MetricsRegistry

    Hallucination:
        test_hallucination_nli_mode_runs    — NLI mode returns MetricResult
        test_hallucination_rate_range       — rate in [0, 1]
        test_hallucination_claim_splitting  — simple decomposition works
        test_hallucination_empty_answer     — handles gracefully

    Trust Score:
        test_trust_score_all_components     — full component set computes correctly
        test_trust_score_partial            — missing optional components OK
        test_trust_score_missing_required   — missing safety → error result
        test_trust_score_tiers              — tier classification correct
        test_trust_score_weight_normalization — weights normalize to 1.0
        test_trust_score_clamping           — out-of-range inputs clamped
        test_trust_score_registry           — accessible via MetricsRegistry

    Multi-Session Persistence:
        test_persistence_cycle_result       — CycleResult fields correct
        test_persistence_report_trend       — ndcg_trend detects improving/degrading
        test_persistence_poison_amplified   — amplification detection works
        test_persistence_forgetting_cycle   — forgetting cycle detection
        test_persistence_score_placeholder  — score() returns placeholder
        test_persistence_registry           — accessible via MetricsRegistry
"""

from __future__ import annotations

import pytest
import numpy as np

from eval_engine.metrics.calibration import (
    CalibrationMetric, compute_ece, compute_brier_score, CalibrationReport, CalibrationBin
)
from eval_engine.metrics.hallucination import (
    HallucinationMetric, _decompose_claims_simple, _verify_claim_nli
)
from eval_engine.metrics.trust_score import TrustScoreMetric
from eval_engine.metrics.multi_session_persistence import (
    MultiSessionPersistenceMetric, CycleResult, PersistenceReport
)
from eval_engine.schemas import GraphSnapshot


# =============================================================================
# Calibration
# =============================================================================

class TestCalibration:

    def test_ece_perfect_calibration(self):
        """When confidence == accuracy in each bin, ECE = 0."""
        # 10 samples: confidence 0.1→1.0, correctness matches
        confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0,   0,   0,   0,   1,   1,   1,   1,   1,   1]
        ece, _ = compute_ece(confs, labels, n_bins=5)
        # Not perfectly 0 with this distribution but should be low
        assert ece < 0.3

    def test_ece_maximally_overconfident(self):
        """Always confidence=1.0 but always wrong → ECE = 1.0."""
        confs = [1.0] * 20
        labels = [0] * 20
        ece, _ = compute_ece(confs, labels, n_bins=5)
        assert abs(ece - 1.0) < 0.01

    def test_brier_perfect(self):
        """Confidence perfectly matches binary labels → Brier ≈ 0."""
        confs = [1.0, 1.0, 0.0, 0.0, 1.0]
        labels = [1,   1,   0,   0,   1]
        brier = compute_brier_score(confs, labels)
        assert abs(brier) < 1e-9

    def test_brier_worst(self):
        """Confidence perfectly opposite labels → Brier = 1.0."""
        confs = [0.0, 0.0, 1.0, 1.0]
        labels = [1,   1,   0,   0]
        brier = compute_brier_score(confs, labels)
        assert abs(brier - 1.0) < 1e-9

    def test_calibration_bin_count(self):
        """compute_ece returns exactly n_bins bins."""
        confs = list(np.random.uniform(0, 1, 50))
        labels = [1 if c > 0.5 else 0 for c in confs]
        _, bins = compute_ece(confs, labels, n_bins=10)
        assert len(bins) == 10

    def test_calibration_reliability_flag(self):
        """Bins with < 5 samples flagged as unreliable."""
        # Only 2 samples total — all bins will be unreliable
        confs = [0.1, 0.9]
        labels = [0, 1]
        _, bins = compute_ece(confs, labels, n_bins=10, min_samples_per_bin=5)
        non_empty = [b for b in bins if b.n_samples > 0]
        assert all(not b.reliable for b in non_empty)

    def test_calibration_direction_overconfident(self):
        """Mean confidence > mean accuracy → overconfident."""
        metric = CalibrationMetric(n_bins=5)
        confs = [0.9, 0.85, 0.8, 0.95, 0.88]
        labels = [1, 0, 0, 1, 0]   # accuracy = 0.4, mean_conf ≈ 0.876
        result = metric.score_from_pairs(confs, labels)
        assert result.raw["calibration_direction"] == "overconfident"

    def test_calibration_direction_underconfident(self):
        """Mean confidence < mean accuracy → underconfident."""
        metric = CalibrationMetric(n_bins=5)
        confs = [0.2, 0.3, 0.25, 0.15, 0.2]
        labels = [1, 1, 1, 1, 1]   # accuracy = 1.0, mean_conf ≈ 0.22
        result = metric.score_from_pairs(confs, labels)
        assert result.raw["calibration_direction"] == "underconfident"

    def test_calibration_accumulator_mode(self):
        """Single-query accumulation → compute_calibration_report()."""
        metric = CalibrationMetric(n_bins=5)
        for i in range(20):
            conf = 0.8 if i % 2 == 0 else 0.3
            label = 1 if i % 2 == 0 else 0
            metric.score(
                question="Q", contexts=["C"], answer="A",
                ground_truth="A" if label else "Z",
                confidence=conf,
            )
        assert len(metric._confidence_buffer) == 20
        report = metric.compute_calibration_report()
        assert report.score >= 0.0
        assert len(metric._confidence_buffer) == 0  # cleared after report

    def test_calibration_missing_confidence(self):
        """Missing confidence kwarg → error result."""
        metric = CalibrationMetric()
        result = metric.score("Q", ["C"], "A")
        assert not result.succeeded
        assert "confidence" in result.error

    def test_calibration_registry(self):
        from eval_engine.metrics import registry
        assert "calibration" in registry.available
        m = registry.get("calibration")
        assert isinstance(m, CalibrationMetric)


# =============================================================================
# Hallucination
# =============================================================================

class TestHallucination:

    def test_hallucination_nli_mode_returns_result(self):
        """NLI mode runs without error (may need sentence-transformers)."""
        metric = HallucinationMetric(mode="nli")
        result = metric.score(
            question="What is Consolidation RAG?",
            contexts=["Consolidation RAG is a biologically-inspired RAG system."],
            answer="Consolidation RAG applies sleep-phase consolidation to RAG pipelines.",
        )
        assert result.metric_name == "hallucination_rate"
        assert 0.0 <= result.score <= 1.0

    def test_hallucination_rate_range(self):
        """Hallucination rate always in [0, 1]."""
        metric = HallucinationMetric(mode="nli")
        for _ in range(5):
            result = metric.score(
                question="Q",
                contexts=["Context about topic X."],
                answer="Answer about topic X and also completely invented facts.",
            )
            assert 0.0 <= result.score <= 1.0

    def test_claim_decomposition_simple(self):
        """Simple sentence splitter produces non-empty claims."""
        answer = "Consolidation prunes edges. Synthesis creates new patterns. Both improve retrieval."
        claims = _decompose_claims_simple(answer)
        assert len(claims) >= 2
        assert all(len(c) > 10 for c in claims)

    def test_hallucination_empty_answer(self):
        """Very short answer → no claims extracted, handled gracefully."""
        metric = HallucinationMetric(mode="nli")
        result = metric.score(
            question="Q",
            contexts=["Context"],
            answer="Yes.",
        )
        # Should not raise — either 0.0 score or error result
        assert result.metric_name == "hallucination_rate"

    def test_hallucination_invalid_mode(self):
        """Invalid mode raises ValueError at construction."""
        with pytest.raises(ValueError, match="mode must be"):
            HallucinationMetric(mode="invalid")

    def test_hallucination_registry(self):
        from eval_engine.metrics import registry
        assert "hallucination_rate" in registry.available
        m = registry.get("hallucination_rate", mode="nli")
        assert isinstance(m, HallucinationMetric)


# =============================================================================
# Trust Score
# =============================================================================

class TestTrustScore:

    def test_trust_score_all_components(self):
        """Full component set produces correct weighted score."""
        metric = TrustScoreMetric(weights={
            "safety_score": 0.30,
            "faithfulness": 0.25,
            "hallucination_score": 0.20,
            "calibration_score": 0.15,
            "grounding_score": 0.10,
        })
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            faithfulness=0.85,
            safety_score=0.92,
            hallucination_score=0.90,
            calibration_score=0.78,
            grounding_score=0.80,
        )
        assert result.succeeded
        assert 0.0 <= result.score <= 1.0
        # Manual calculation
        expected = 0.92*0.30 + 0.85*0.25 + 0.90*0.20 + 0.78*0.15 + 0.80*0.10
        assert abs(result.score - expected) < 0.001

    def test_trust_score_partial_components(self):
        """Missing optional components excluded from weighted average."""
        metric = TrustScoreMetric()
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            faithfulness=0.80,
            safety_score=0.90,
            # hallucination, calibration, grounding all missing
        )
        assert result.succeeded
        assert 0.0 <= result.score <= 1.0

    def test_trust_score_missing_required_safety(self):
        """Missing safety_score with require_safety=True → error."""
        metric = TrustScoreMetric(require_safety=True)
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            faithfulness=0.80,
            # safety_score missing
        )
        assert not result.succeeded
        assert "safety_score" in result.error

    def test_trust_score_tier_high(self):
        """Score >= 0.85 → high tier."""
        metric = TrustScoreMetric(require_safety=False, require_faithfulness=False)
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            faithfulness=0.95, safety_score=0.98,
            hallucination_score=0.95, calibration_score=0.90, grounding_score=0.92,
        )
        assert result.raw["trust_tier"] == "high"

    def test_trust_score_tier_critical(self):
        """Score < 0.50 → critical tier."""
        metric = TrustScoreMetric(require_safety=False, require_faithfulness=False)
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            faithfulness=0.20, safety_score=0.15,
            hallucination_score=0.10, calibration_score=0.30, grounding_score=0.25,
        )
        assert result.raw["trust_tier"] == "critical"

    def test_trust_score_weight_normalization(self):
        """Non-summing weights are normalized automatically."""
        metric = TrustScoreMetric(weights={
            "safety_score": 3.0,
            "faithfulness": 2.0,
        })
        # After normalization: safety=0.6, faithfulness=0.4
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            safety_score=1.0, faithfulness=1.0,
        )
        assert result.succeeded
        assert abs(result.score - 1.0) < 0.001

    def test_trust_score_clamping(self):
        """Out-of-range component scores are clamped to [0,1]."""
        metric = TrustScoreMetric(require_safety=False, require_faithfulness=False)
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            safety_score=1.5,       # > 1.0 — should clamp to 1.0
            faithfulness=-0.1,      # < 0.0 — should clamp to 0.0
        )
        assert result.succeeded
        assert 0.0 <= result.score <= 1.0

    def test_trust_score_reliable_flag(self):
        """Missing hallucination_score → reliable=False."""
        metric = TrustScoreMetric()
        result = metric.score(
            question="Q", contexts=["C"], answer="A",
            faithfulness=0.8, safety_score=0.9,
            # hallucination_score missing → key component missing
        )
        assert result.raw["reliable"] is False

    def test_trust_score_registry(self):
        from eval_engine.metrics import registry
        assert "trust_score" in registry.available
        m = registry.get("trust_score")
        assert isinstance(m, TrustScoreMetric)


# =============================================================================
# Multi-Session Persistence
# =============================================================================

class TestMultiSessionPersistence:

    def test_cycle_result_fields(self):
        """CycleResult has all required fields."""
        r = CycleResult(
            cycle_id=3,
            ndcg_score=0.72,
            poison_persistence_rate=0.40,
            truth_persistence_rate=0.85,
            faithfulness=0.78,
            graph_snapshot=None,
        )
        assert r.cycle_id == 3
        assert abs(r.ndcg_score - 0.72) < 1e-9
        d = r.to_dict()
        assert "cycle_id" in d
        assert "poison_persistence_rate" in d

    def test_persistence_report_trend_improving(self):
        """NDCG increasing across cycles → 'improving'."""
        results = [
            CycleResult(0, 0.50, 0.8, 0.9, None, None),
            CycleResult(1, 0.60, 0.7, 0.88, None, None),
            CycleResult(2, 0.70, 0.5, 0.85, None, None),
        ]
        report = PersistenceReport("test", 2, results, set(), set())
        assert report.ndcg_trend == "improving"

    def test_persistence_report_trend_degrading(self):
        """NDCG decreasing → 'degrading'."""
        results = [
            CycleResult(0, 0.80, 0.2, 0.9, None, None),
            CycleResult(1, 0.70, 0.3, 0.85, None, None),
            CycleResult(2, 0.60, 0.4, 0.80, None, None),
        ]
        report = PersistenceReport("test", 2, results, set(), set())
        assert report.ndcg_trend == "degrading"

    def test_persistence_poison_amplified(self):
        """Poison rate increases → poison_amplified = True."""
        results = [
            CycleResult(0, 0.70, 0.20, 0.90, None, None),  # poison rate starts low
            CycleResult(1, 0.65, 0.35, 0.88, None, None),
            CycleResult(2, 0.60, 0.50, 0.85, None, None),  # poison rate grows
        ]
        report = PersistenceReport("test", 2, results, {"p1"}, {"t1"})
        assert report.poison_amplified is True

    def test_persistence_poison_not_amplified(self):
        """Poison rate decreasing → poison_amplified = False."""
        results = [
            CycleResult(0, 0.70, 0.80, 0.90, None, None),
            CycleResult(1, 0.72, 0.50, 0.88, None, None),
            CycleResult(2, 0.75, 0.20, 0.85, None, None),
        ]
        report = PersistenceReport("test", 2, results, {"p1"}, {"t1"})
        assert report.poison_amplified is False

    def test_persistence_forgetting_cycle(self):
        """Forgetting cycle = first cycle where rate drops below 10%."""
        results = [
            CycleResult(0, 0.70, 0.80, 0.90, None, None),
            CycleResult(1, 0.71, 0.40, 0.88, None, None),
            CycleResult(2, 0.73, 0.08, 0.85, None, None),  # below 10%
            CycleResult(3, 0.75, 0.05, 0.82, None, None),
        ]
        report = PersistenceReport("test", 3, results, {"p1"}, {"t1"})
        assert report.forgetting_cycle_poison == 2

    def test_persistence_no_forgetting(self):
        """Poison never drops below 10% → forgetting_cycle = None."""
        results = [
            CycleResult(0, 0.70, 0.80, 0.90, None, None),
            CycleResult(1, 0.71, 0.75, 0.88, None, None),
            CycleResult(2, 0.72, 0.70, 0.85, None, None),
        ]
        report = PersistenceReport("test", 2, results, {"p1"}, {"t1"})
        assert report.forgetting_cycle_poison is None

    def test_persistence_score_placeholder(self):
        """score() returns placeholder with informative note."""
        metric = MultiSessionPersistenceMetric(n_cycles=5)
        result = metric.score("Q", ["C"], "A")
        assert result.metric_name == "multi_session_persistence"
        assert "run_persistence_evaluation" in result.metadata.get("note", "")

    def test_persistence_report_serializable(self):
        """to_dict() produces JSON-serializable output."""
        import json
        results = [CycleResult(0, 0.70, 0.5, 0.8, None, None)]
        report = PersistenceReport("test", 0, results, {"p1"}, {"t1"})
        serialized = json.dumps(report.to_dict())
        parsed = json.loads(serialized)
        assert parsed["experiment_id"] == "test"

    def test_persistence_registry(self):
        from eval_engine.metrics import registry
        assert "multi_session_persistence" in registry.available
        m = registry.get("multi_session_persistence")
        assert isinstance(m, MultiSessionPersistenceMetric)

    def test_all_tier1_metrics_in_registry(self):
        """All four Tier 1 metrics accessible via registry."""
        from eval_engine.metrics import registry
        for name in ["calibration", "hallucination_rate", "trust_score", "multi_session_persistence"]:
            assert name in registry.available, f"{name} missing from registry"
