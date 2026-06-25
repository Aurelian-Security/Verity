"""
eval_engine/tests/test_phase2_metrics.py

Phase 2 unit tests — Consolidation RAG Evaluation Tests 2-5.

Covers:
    Per-Stage Ablation (Test 2):
        test_ablation_report_structure
        test_ablation_critical_stage_detection
        test_ablation_contribution_pct

    Threshold Compute Budget (Test 3):
        test_budget_curve_baseline
        test_budget_curve_inflection_detection
        test_budget_min_effective_budget

    Single-Session Poisoning (Test 4):
        test_poison_corpus_builder_rate
        test_poison_corpus_builder_indices
        test_poisoning_result_degradation
        test_poisoning_report_asr
        test_poisoning_report_fnr

    Query Perturbation (Test 5):
        test_typo_noise_modifies_query
        test_truncation_shortens_query
        test_expansion_prepends_query
        test_paraphrase_changes_opener
        test_perturbation_robustness_score_range
        test_perturbation_report_most_vulnerable
"""

from __future__ import annotations

import numpy as np

from eval_engine.schemas import RetrievalCase


# =============================================================================
# Per-Stage Ablation (Test 2)
# =============================================================================

class TestPerStageAblation:

    def _make_cases(self) -> list[RetrievalCase]:
        return [
            RetrievalCase(query_id=f"q{i}", query=f"query {i}", relevant_ids=frozenset([f"doc_{i}"]))
            for i in range(5)
        ]

    def _relevance_map(self, cases: list[RetrievalCase]) -> dict[str, dict[str, float]]:
        return {c.query_id: {f"doc_{i}": 1.0} for i, c in enumerate(cases)}

    def _perfect_retriever(self, query: str) -> list[str]:
        idx = query.split()[-1]
        return [f"doc_{idx}", "doc_x", "doc_y"]

    def _degraded_retriever(self, query: str) -> list[str]:
        """Returns wrong docs — simulates ablated stage hurting retrieval."""
        return ["doc_wrong1", "doc_wrong2", "doc_wrong3"]

    def test_ablation_report_structure(self):
        from eval_engine.metrics.per_stage_ablation import PerStageAblationMetric

        cases = self._make_cases()
        rel = self._relevance_map(cases)
        metric = PerStageAblationMetric(k=3)

        report = metric.run_ablation(
            cases=cases,
            stage_retrievers={
                1: ("stage_1", self._degraded_retriever),
                2: ("stage_2", self._perfect_retriever),
            },
            full_pipeline_retriever=self._perfect_retriever,
            relevance_by_query=rel,
            experiment_id="test_ablation",
        )

        assert report.experiment_id == "test_ablation"
        assert len(report.stage_results) == 2
        assert report.full_pipeline_score > 0

    def test_ablation_critical_stage_detection(self):
        from eval_engine.metrics.per_stage_ablation import PerStageAblationMetric

        cases = self._make_cases()
        rel = self._relevance_map(cases)
        metric = PerStageAblationMetric(k=3)

        report = metric.run_ablation(
            cases=cases,
            stage_retrievers={
                1: ("critical_stage", self._degraded_retriever),   # ablating this hurts a lot
                2: ("redundant_stage", self._perfect_retriever),   # ablating this doesn't hurt
            },
            full_pipeline_retriever=self._perfect_retriever,
            relevance_by_query=rel,
        )

        stage_1_result = next(s for s in report.stage_results if s.stage_id == 1)
        stage_2_result = next(s for s in report.stage_results if s.stage_id == 2)

        # Stage 1 ablation (degraded retriever) should have positive delta
        assert stage_1_result.delta > 0, "Ablating critical stage should degrade score"
        # Stage 2 ablation (perfect retriever) should have near-zero delta
        assert stage_2_result.delta <= stage_1_result.delta

    def test_ablation_contribution_pct(self):
        from eval_engine.metrics.per_stage_ablation import AblationStageResult

        result = AblationStageResult(
            stage_id=1, stage_name="nrem",
            score=0.5, full_pipeline_score=0.8,
            delta=0.3,
        )
        assert abs(result.contribution_pct - 37.5) < 0.01
        assert result.is_critical   # 37.5% > 10% threshold

    def test_ablation_ranked(self):
        from eval_engine.metrics.per_stage_ablation import AblationReport, AblationStageResult

        s1 = AblationStageResult(stage_id=1, stage_name="a", score=0.5, full_pipeline_score=0.8, delta=0.3)
        s2 = AblationStageResult(stage_id=2, stage_name="b", score=0.75, full_pipeline_score=0.8, delta=0.05)
        report = AblationReport(experiment_id="test", full_pipeline_score=0.8, stage_results=[s2, s1])

        ranked = report.ranked()
        assert ranked[0].stage_id == 1   # Highest delta first
        assert ranked[1].stage_id == 2


# =============================================================================
# Threshold Compute Budget (Test 3)
# =============================================================================

class TestThresholdComputeBudget:

    def _make_cases(self) -> list[RetrievalCase]:
        return [
            RetrievalCase(query_id=f"q{i}", query=f"query {i}", relevant_ids=frozenset([f"doc_{i}"]))
            for i in range(4)
        ]

    def _relevance(self, cases) -> dict[str, dict[str, float]]:
        return {c.query_id: {f"doc_{i}": 1.0} for i, c in enumerate(cases)}

    def test_budget_curve_baseline(self):
        from eval_engine.metrics.threshold_compute_budget import ThresholdComputeBudgetMetric

        cases = self._make_cases()
        rel = self._relevance(cases)
        metric = ThresholdComputeBudgetMetric(k=3)

        def baseline_retriever(q: str) -> list[str]:
            return ["doc_wrong"]

        def budget_retriever(q: str, budget: float) -> list[str]:
            idx = q.split()[-1]
            if budget >= 0.5:
                return [f"doc_{idx}"]
            return ["doc_wrong"]

        curve = metric.run_budget_sweep(
            cases=cases,
            baseline_retriever=baseline_retriever,
            budget_retriever=budget_retriever,
            budget_levels=[0.25, 0.50, 0.75, 1.00],
            relevance_by_query=rel,
        )

        assert curve.baseline_score == 0.0   # baseline gets nothing right
        assert len(curve.points) == 4

    def test_budget_curve_inflection_detection(self):
        from eval_engine.metrics.threshold_compute_budget import _find_inflection_point

        levels = [0.25, 0.50, 0.75, 1.00]
        scores = [0.30, 0.65, 0.68, 0.69]  # big jump at 0.5, then plateau

        inflection, plateau = _find_inflection_point(levels, scores, plateau_threshold=0.05)
        assert inflection == 0.75   # first point where improvement < 0.05
        assert plateau == 0.68

    def test_budget_min_effective_budget(self):
        from eval_engine.metrics.threshold_compute_budget import BudgetCurveResult, BudgetPoint

        points = [
            BudgetPoint(0.25, "25%", ndcg_score=0.50, delta_from_baseline=0.0),
            BudgetPoint(0.50, "50%", ndcg_score=0.63, delta_from_baseline=0.13),  # > 5% over 0.60 baseline
            BudgetPoint(0.75, "75%", ndcg_score=0.70, delta_from_baseline=0.10),
        ]
        curve = BudgetCurveResult(
            experiment_id="test", baseline_score=0.60,
            points=points, inflection_budget=0.75, plateau_score=0.70,
        )
        # Min effective = first budget where score >= 0.60 * 1.05 = 0.63
        assert curve.min_effective_budget == 0.50


# =============================================================================
# Single-Session Poisoning (Test 4)
# =============================================================================

class TestSingleSessionPoisoning:

    def test_poison_corpus_builder_rate(self):
        from eval_engine.metrics.single_session_poisoning import PoisonCorpusBuilder

        builder = PoisonCorpusBuilder(seed=42)
        clean = [f"Document {i} about topic {i}." for i in range(20)]
        poisoned, indices = builder.build_poison_set(clean, poison_rate=0.1)

        assert len(poisoned) == 20           # corpus size unchanged
        assert len(indices) == 2             # 10% of 20 = 2 docs poisoned
        assert all(isinstance(i, int) for i in indices)

    def test_poison_corpus_builder_indices_differ(self):
        """Poisoned docs differ from originals."""
        from eval_engine.metrics.single_session_poisoning import PoisonCorpusBuilder

        builder = PoisonCorpusBuilder(seed=42)
        clean = [f"Document {i}." for i in range(10)]
        poisoned, indices = builder.build_poison_set(clean, poison_rate=0.2)

        for idx in indices:
            assert poisoned[idx] != clean[idx], f"Doc {idx} should be modified"

    def test_poisoning_result_degradation(self):
        from eval_engine.metrics.single_session_poisoning import PoisoningResult

        result = PoisoningResult(
            query_id="q1", attack_type="factual_substitution",
            poison_rate=0.1, clean_score=0.8, poisoned_score=0.5,
            faithfulness_clean=0.9, faithfulness_poisoned=0.6,
        )
        assert abs(result.ndcg_degradation - 0.3) < 1e-9
        assert abs(result.faithfulness_degradation - 0.3) < 1e-9

    def test_poisoning_report_asr(self):
        from eval_engine.metrics.single_session_poisoning import PoisoningReport, PoisoningResult

        results = [
            PoisoningResult("q1", "factual", 0.1, 0.8, 0.5, attack_succeeded=True),
            PoisoningResult("q2", "factual", 0.1, 0.8, 0.7, attack_succeeded=False),
            PoisoningResult("q3", "factual", 0.1, 0.8, 0.6, attack_succeeded=True),
            PoisoningResult("q4", "factual", 0.1, 0.8, 0.8, attack_succeeded=False),
        ]
        report = PoisoningReport("test", "factual", 0.1, results)
        assert abs(report.attack_success_rate - 0.5) < 1e-9

    def test_poisoning_report_fnr(self):
        """FNR = attacks that succeeded but LlamaGuard missed."""
        from eval_engine.metrics.single_session_poisoning import PoisoningReport, PoisoningResult

        results = [
            PoisoningResult("q1", "factual", 0.1, 0.8, 0.5,
                            attack_succeeded=True, llamaguard_detected=False),  # missed
            PoisoningResult("q2", "factual", 0.1, 0.8, 0.6,
                            attack_succeeded=True, llamaguard_detected=True),   # caught
        ]
        report = PoisoningReport("test", "factual", 0.1, results)
        assert abs(report.false_negative_rate - 0.5) < 1e-9  # 1 of 2 attacks missed


# =============================================================================
# Query Perturbation (Test 5)
# =============================================================================

class TestQueryPerturbation:

    def test_typo_noise_modifies_query(self):
        from eval_engine.metrics.query_perturbation import _typo_noise
        original = "What does the consolidation phase do in a memory-augmented RAG system?"
        perturbed = _typo_noise(original, noise_rate=0.15, seed=42)
        assert perturbed != original
        assert len(perturbed) > 0

    def test_truncation_shortens_query(self):
        from eval_engine.metrics.query_perturbation import _truncate
        original = "What does the consolidation phase do in a memory-augmented RAG system?"
        truncated = _truncate(original, keep_fraction=0.5)
        assert len(truncated.split()) < len(original.split())

    def test_expansion_prepends_context(self):
        from eval_engine.metrics.query_perturbation import _expand
        original = "What does the consolidation phase improve?"
        expanded = _expand(original)
        assert original in expanded
        assert len(expanded) > len(original)

    def test_paraphrase_changes_opener(self):
        from eval_engine.metrics.query_perturbation import _paraphrase_simple
        original = "What is Consolidation RAG?"
        paraphrased = _paraphrase_simple(original)
        assert paraphrased != original
        assert "Consolidation RAG" in paraphrased

    def test_perturbation_robustness_score_range(self):
        from eval_engine.metrics.query_perturbation import PerturbationTypeResult

        result = PerturbationTypeResult(
            perturbation_type="typo_noise",
            mean_ndcg_clean=0.80,
            mean_ndcg_perturbed=0.65,
            robustness_score=0.65 / 0.80,
            score_variance=0.02,
            n_queries=10,
        )
        assert 0.0 <= result.robustness_score <= 1.0

    def test_perturbation_report_most_vulnerable(self):
        from eval_engine.metrics.query_perturbation import PerturbationReport, PerturbationTypeResult

        results = [
            PerturbationTypeResult("typo_noise", 0.8, 0.7, 0.875, 0.01, 10),
            PerturbationTypeResult("truncation", 0.8, 0.4, 0.5, 0.02, 10),     # lowest PRS
            PerturbationTypeResult("paraphrase", 0.8, 0.75, 0.9375, 0.005, 10),
        ]
        report = PerturbationReport("test", k=10, type_results=results)
        assert report.most_vulnerable_type == "truncation"
        assert abs(report.overall_robustness_score - np.mean([0.875, 0.5, 0.9375])) < 1e-6

    def test_perturbation_metric_registry(self):
        """All Phase 2 metrics importable via registry."""
        from eval_engine.metrics import registry
        available = registry.available
        assert "per_stage_ablation" in available
        assert "threshold_compute_budget" in available
        assert "single_session_poisoning" in available
        assert "query_perturbation" in available
