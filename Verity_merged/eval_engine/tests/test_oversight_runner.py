"""
eval_engine/tests/test_oversight_runner.py

Phase 3 Completion — OversightRunner tests.

All tests use dry_run=True — zero API cost.

Covers:
    test_oversight_runner_completes         — full dataset runs end to end
    test_oversight_result_fields            — OversightRunResult has all fields
    test_oversight_verdict_distribution     — verdict counts correct
    test_oversight_reward_hacking_rate      — RH rate computes correctly
    test_oversight_mean_scores              — mean safety/accuracy correct
    test_oversight_jsonl_output             — results.jsonl written
    test_oversight_manifest_output          — manifest.json written
    test_oversight_debate_traces            — individual debate JSONs saved
    test_oversight_stat_report             — statistical analysis runs
    test_oversight_cli_dry_run              — CLI oversight-run --dry-run works
    test_oversight_runner_exported          — importable from eval_engine
"""

from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path


SAMPLE_DATASET = [
    {
        "query_id": f"q{i:03d}",
        "question": f"What does the consolidation phase do for query {i}?",
        "contexts": [
            f"Context A for query {i}: The consolidation phase improves retrieval.",
            f"Context B for query {i}: Graph pruning removes low-weight edges.",
        ],
    }
    for i in range(5)
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_config(tmp_path):
    """Minimal EvalConfig for oversight testing."""
    from eval_engine.config import (
        EvalConfig, DatasetConfig, BudgetConfig, AsyncConfig, StatisticsConfig
    )

    # Write a dummy dataset file
    ds_path = tmp_path / "eval_set.json"
    ds_path.write_text(json.dumps(SAMPLE_DATASET))

    return EvalConfig(
        experiment_id="test_oversight_run",
        architecture="centralized",
        model="claude-haiku-4-5",
        dataset=DatasetConfig(path=ds_path),
        metrics=[],
        budget=BudgetConfig(max_usd=5.00),
        async_cfg=AsyncConfig(max_concurrent_queries=2, batch_size=3),
        statistics=StatisticsConfig(alpha=0.05),
        output_dir=tmp_path / "outputs",
    )


@pytest.fixture
def oversight_runner(eval_config):
    from eval_engine.orchestration.oversight_runner import OversightRunner
    return OversightRunner(
        config=eval_config,
        dry_run=True,
        use_celery=False,
    )


# ---------------------------------------------------------------------------
# OversightRunResult unit tests
# ---------------------------------------------------------------------------

class TestOversightRunResult:

    def test_result_fields(self):
        from eval_engine.orchestration.oversight_runner import OversightRunResult
        result = OversightRunResult(experiment_id="test", dry_run=True)
        assert result.experiment_id == "test"
        assert result.dry_run is True
        assert result.debate_results == []
        assert result.errors == []

    def test_verdict_distribution(self):
        from eval_engine.orchestration.oversight_runner import OversightRunResult
        from eval_engine.orchestration.debate_round import DebateResult
        from eval_engine.agents.agent_base import AgentTrace, AgentOutput

        def _dummy_output(role):
            trace = AgentTrace(
                agent_role=role, model="mock",
                prompt_preview="", response_preview="",
                prompt_tokens=0, completion_tokens=0,
                latency_seconds=0.0, dry_run=True,
            )
            return AgentOutput(agent_role=role, content="mock", trace=trace)

        result = OversightRunResult(experiment_id="test")
        for verdict in ["Pass", "Pass", "Fail", "Conditional"]:
            dr = DebateResult(
                query_id="q1", query="Q", n_contexts=2,
                proposal=_dummy_output("proposer"),
                critique=_dummy_output("critic"),
                judgment=_dummy_output("judge"),
                final_safety_score=0.9,
                final_accuracy_score=0.8,
                verdict=verdict,
                reward_hacking_confirmed=False,
                total_latency_seconds=0.1,
                dry_run=True,
            )
            result.add_result(dr)

        dist = result.verdict_distribution
        assert dist["Pass"] == 2
        assert dist["Fail"] == 1
        assert dist["Conditional"] == 1

    def test_reward_hacking_rate(self):
        from eval_engine.orchestration.oversight_runner import OversightRunResult
        from eval_engine.orchestration.debate_round import DebateResult
        from eval_engine.agents.agent_base import AgentTrace, AgentOutput

        def _dummy_output(role):
            trace = AgentTrace(
                agent_role=role, model="mock",
                prompt_preview="", response_preview="",
                prompt_tokens=0, completion_tokens=0,
                latency_seconds=0.0, dry_run=True,
            )
            return AgentOutput(agent_role=role, content="mock", trace=trace)

        result = OversightRunResult(experiment_id="test")
        for rh in [True, False, True, False, False]:
            dr = DebateResult(
                query_id="q1", query="Q", n_contexts=2,
                proposal=_dummy_output("proposer"),
                critique=_dummy_output("critic"),
                judgment=_dummy_output("judge"),
                final_safety_score=0.9,
                final_accuracy_score=0.8,
                verdict="Pass",
                reward_hacking_confirmed=rh,
                total_latency_seconds=0.1,
                dry_run=True,
            )
            result.add_result(dr)

        assert abs(result.reward_hacking_rate - 0.4) < 1e-9

    def test_manifest_keys(self):
        from eval_engine.orchestration.oversight_runner import OversightRunResult
        result = OversightRunResult(experiment_id="test")
        result.finalize()
        manifest = result.to_manifest()
        required = [
            "experiment_id", "pipeline", "dry_run",
            "duration_seconds", "total_debates", "total_errors",
            "success_rate", "reward_hacking_rate", "verdict_distribution",
        ]
        for key in required:
            assert key in manifest, f"Missing key: {key}"

    def test_pipeline_label(self):
        from eval_engine.orchestration.oversight_runner import OversightRunResult
        result = OversightRunResult(experiment_id="test")
        result.finalize()
        assert result.to_manifest()["pipeline"] == "oversight"


# ---------------------------------------------------------------------------
# OversightRunner integration tests
# ---------------------------------------------------------------------------

class TestOversightRunner:

    def test_runner_completes(self, oversight_runner, eval_config):
        """Full dataset runs end to end in dry-run mode."""
        result = asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        assert len(result.debate_results) == len(SAMPLE_DATASET)
        assert result.errors == []
        assert result.success_rate == 1.0

    def test_runner_result_has_scores(self, oversight_runner):
        """DebateResults have safety and accuracy scores."""
        result = asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        for dr in result.debate_results:
            assert 0.0 <= dr.final_safety_score <= 1.0
            assert 0.0 <= dr.final_accuracy_score <= 1.0

    def test_runner_result_has_verdicts(self, oversight_runner):
        """All debates have valid verdicts."""
        result = asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        valid_verdicts = {"Pass", "Conditional", "Fail"}
        for dr in result.debate_results:
            assert dr.verdict in valid_verdicts

    def test_runner_writes_jsonl(self, oversight_runner, eval_config):
        """oversight_results.jsonl written after run."""
        asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        results_path = eval_config.output_dir / eval_config.experiment_id / "oversight_results.jsonl"
        assert results_path.exists()
        lines = results_path.read_text().strip().split("\n")
        assert len(lines) == len(SAMPLE_DATASET)

    def test_runner_writes_manifest(self, oversight_runner, eval_config):
        """oversight_manifest.json written after run."""
        asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        manifest_path = eval_config.output_dir / eval_config.experiment_id / "oversight_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["experiment_id"] == eval_config.experiment_id
        assert data["pipeline"] == "oversight"
        assert data["total_debates"] == len(SAMPLE_DATASET)

    def test_runner_writes_debate_traces(self, oversight_runner, eval_config):
        """Individual debate JSON files written to debates/ subdirectory."""
        asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        debates_dir = eval_config.output_dir / eval_config.experiment_id / "debates"
        assert debates_dir.exists()
        debate_files = list(debates_dir.glob("debate_*.json"))
        assert len(debate_files) == len(SAMPLE_DATASET)

    def test_runner_stat_report(self, oversight_runner, eval_config):
        """Statistical analysis runs and saves oversight_stats.json."""
        asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        stats_path = eval_config.output_dir / eval_config.experiment_id / "oversight_stats.json"
        assert stats_path.exists()
        data = json.loads(stats_path.read_text())
        assert "results" in data

    def test_runner_dry_run_flag(self, oversight_runner):
        """dry_run=True reflected in OversightRunResult."""
        result = asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        assert result.dry_run is True

    def test_runner_mean_scores(self, oversight_runner):
        """Mean safety and accuracy scores are valid floats."""
        result = asyncio.run(oversight_runner.run(SAMPLE_DATASET))
        safety = result.mean_safety_score()
        accuracy = result.mean_accuracy_score()
        assert safety is not None
        assert accuracy is not None
        assert 0.0 <= safety <= 1.0
        assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestOversightCLI:

    def test_oversight_run_command_exists(self):
        """verity oversight-run command is registered."""
        from typer.testing import CliRunner
        from eval_engine.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["oversight-run", "--help"])
        assert result.exit_code == 0
        assert "oversight" in result.output.lower()

    def test_oversight_run_dry_run_no_dataset(self):
        """oversight-run without --dataset shows error."""
        from typer.testing import CliRunner
        from eval_engine.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["oversight-run", "--dry-run"])
        assert result.exit_code != 0 or "required" in result.output.lower() or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestPhase3Exports:

    def test_oversight_runner_importable(self):
        """OversightRunner importable from eval_engine."""
        from eval_engine import OversightRunner
        assert OversightRunner is not None

    def test_oversight_run_result_importable(self):
        """OversightRunResult importable from eval_engine."""
        from eval_engine import OversightRunResult
        assert OversightRunResult is not None

    def test_debate_result_importable(self):
        """DebateResult importable from eval_engine."""
        from eval_engine import DebateResult
        assert DebateResult is not None

    def test_debate_round_importable(self):
        """DebateRound importable from eval_engine."""
        from eval_engine import DebateRound
        assert DebateRound is not None

    def test_orchestration_module_exports(self):
        """orchestration __init__ exports all expected names."""
        from eval_engine.orchestration import (
            DebateRound, DebateResult,
            OversightRunner, OversightRunResult,
            dispatch_debate_batch, collect_debate_results,
        )
        assert all([
            DebateRound, DebateResult,
            OversightRunner, OversightRunResult,
            dispatch_debate_batch, collect_debate_results,
        ])
