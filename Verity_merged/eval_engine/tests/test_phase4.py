"""
eval_engine/tests/test_phase4.py

Phase 4 unit tests — Reproducible Experiment Infrastructure.

Covers:
    ReproducibilityBundle:
        test_bundle_captures_python_version
        test_bundle_captures_packages
        test_bundle_saves_json
        test_bundle_json_has_required_keys
        test_bundle_seed_stored
        test_bundle_from_config
        test_set_global_seed_runs

    DatasetManifest:
        test_manifest_from_records
        test_manifest_dataset_id_is_sha256
        test_manifest_same_data_same_id
        test_manifest_different_data_different_id
        test_manifest_row_count
        test_manifest_columns
        test_manifest_saves_json
        test_manifest_from_file
        test_manifest_lineage_step
        test_manifest_short_id

    Comparison:
        test_compare_loads_manifests
        test_compare_score_deltas
        test_compare_same_dataset_flag
        test_compare_missing_run_returns_empty

    CLI:
        test_manifest_command_exists
        test_compare_command_exists
        test_run_has_seed_flag
        test_run_has_track_flag
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path


# =============================================================================
# ReproducibilityBundle
# =============================================================================

class TestReproducibilityBundle:

    def test_bundle_captures_python_version(self):
        import sys
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="test", seed=42)
        bundle.capture()
        assert sys.version in bundle.python_version

    def test_bundle_captures_packages(self):
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="test", seed=42)
        bundle.capture()
        assert len(bundle.installed_packages) > 0
        assert "pydantic" in bundle.installed_packages

    def test_bundle_saves_json(self, tmp_path):
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="test", seed=42)
        bundle.capture()
        path = bundle.save(tmp_path)
        assert path.exists()
        assert path.name == "reproducibility.json"

    def test_bundle_json_has_required_keys(self, tmp_path):
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="test_run", seed=99)
        bundle.capture()
        bundle.save(tmp_path)
        data = json.loads((tmp_path / "reproducibility.json").read_text())
        required = [
            "experiment_id", "seed", "timestamp", "python_version",
            "platform", "git", "installed_packages", "verity_version",
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_bundle_seed_stored(self, tmp_path):
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="test", seed=1337)
        bundle.capture()
        bundle.save(tmp_path)
        data = json.loads((tmp_path / "reproducibility.json").read_text())
        assert data["seed"] == 1337

    def test_bundle_experiment_id_stored(self, tmp_path):
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="my_experiment", seed=42)
        bundle.capture()
        bundle.save(tmp_path)
        data = json.loads((tmp_path / "reproducibility.json").read_text())
        assert data["experiment_id"] == "my_experiment"

    def test_bundle_from_config(self, tmp_path):
        from eval_engine.config import EvalConfig, DatasetConfig
        from eval_engine.reproducibility import ReproducibilityBundle

        ds_path = tmp_path / "ds.json"
        ds_path.write_text("[]")

        config = EvalConfig(
            experiment_id="config_test",
            architecture="centralized",
            model="claude-haiku-4-5",
            dataset=DatasetConfig(path=ds_path),
            seed=777,
        )
        bundle = ReproducibilityBundle.from_config(config, seed=config.seed)
        assert bundle.experiment_id == "config_test"
        assert bundle.seed == 777

    def test_set_global_seed_runs(self):
        """set_global_seed() runs without error."""
        from eval_engine.reproducibility import set_global_seed
        set_global_seed(42)  # Should not raise

    def test_bundle_git_structure(self, tmp_path):
        """Git section has hash/branch/dirty fields."""
        from eval_engine.reproducibility import ReproducibilityBundle
        bundle = ReproducibilityBundle(experiment_id="test", seed=42)
        bundle.capture()
        bundle.save(tmp_path)
        data = json.loads((tmp_path / "reproducibility.json").read_text())
        assert "hash" in data["git"]
        assert "branch" in data["git"]
        assert "dirty" in data["git"]


# =============================================================================
# DatasetManifest
# =============================================================================

SAMPLE_RECORDS = [
    {"question": f"Q{i}", "answer": f"A{i}", "contexts": [f"C{i}"], "ground_truth": f"GT{i}"}
    for i in range(10)
]


class TestDatasetManifest:

    def test_manifest_from_records(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        assert manifest.row_count == 10
        assert "question" in manifest.columns

    def test_manifest_dataset_id_is_sha256(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        assert manifest.dataset_id.startswith("sha256:")
        assert len(manifest.dataset_id) > 20

    def test_manifest_same_data_same_id(self):
        from eval_engine.dataset_manifest import DatasetManifest
        m1 = DatasetManifest.from_records(SAMPLE_RECORDS)
        m2 = DatasetManifest.from_records(SAMPLE_RECORDS)
        assert m1.dataset_id == m2.dataset_id

    def test_manifest_different_data_different_id(self):
        from eval_engine.dataset_manifest import DatasetManifest
        records_b = [{"question": "Different", "answer": "Data"}]
        m1 = DatasetManifest.from_records(SAMPLE_RECORDS)
        m2 = DatasetManifest.from_records(records_b)
        assert m1.dataset_id != m2.dataset_id

    def test_manifest_row_count(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        assert manifest.row_count == len(SAMPLE_RECORDS)

    def test_manifest_columns(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        assert set(manifest.columns) == {"question", "answer", "contexts", "ground_truth"}

    def test_manifest_saves_json(self, tmp_path):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        path = manifest.save(tmp_path)
        assert path.exists()
        assert path.name == "dataset_manifest.json"

    def test_manifest_json_has_required_keys(self, tmp_path):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        manifest.save(tmp_path)
        data = json.loads((tmp_path / "dataset_manifest.json").read_text())
        required = ["dataset_id", "path", "row_count", "columns", "schema", "lineage"]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_manifest_from_file(self, tmp_path):
        from eval_engine.dataset_manifest import DatasetManifest
        ds_path = tmp_path / "eval_set.json"
        ds_path.write_text(json.dumps(SAMPLE_RECORDS))
        manifest = DatasetManifest.from_file(ds_path)
        assert manifest.row_count == len(SAMPLE_RECORDS)
        assert manifest.dataset_id.startswith("sha256:")
        assert str(ds_path) == manifest.path

    def test_manifest_lineage_step(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        manifest.add_lineage_step(
            "single_session_poisoning",
            output_path="datasets/poisoned.json",
            poison_rate=0.1,
            attack_type="factual_substitution",
        )
        assert len(manifest.lineage) == 1
        assert manifest.lineage[0].step_name == "single_session_poisoning"
        assert manifest.lineage[0].parameters["poison_rate"] == 0.1

    def test_manifest_lineage_chain(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        manifest.add_lineage_step("step_1")
        manifest.add_lineage_step("step_2")
        manifest.add_lineage_step("step_3")
        assert len(manifest.lineage) == 3
        assert manifest.lineage[2].step_name == "step_3"

    def test_manifest_short_id(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        assert manifest.short_id.endswith("...")
        assert len(manifest.short_id) == 15

    def test_manifest_version_tag(self):
        from eval_engine.dataset_manifest import DatasetManifest
        manifest = DatasetManifest.from_records(SAMPLE_RECORDS, version_tag="v1.0")
        assert manifest.version_tag == "v1.0"
        d = manifest.to_dict()
        assert d["version_tag"] == "v1.0"


# =============================================================================
# Comparison
# =============================================================================

def _write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestComparison:

    def test_compare_loads_manifests(self, tmp_path):
        from eval_engine.comparison import compare_run_manifests

        _write_manifest(tmp_path / "run_a" / "manifest.json", {
            "experiment_id": "run_a",
            "mean_scores": {"ndcg": 0.72, "recall_at_k": 0.85},
            "duration_seconds": 45.2,
            "total_records": 100,
            "success_rate": 0.98,
        })
        _write_manifest(tmp_path / "run_b" / "manifest.json", {
            "experiment_id": "run_b",
            "mean_scores": {"ndcg": 0.78, "recall_at_k": 0.88},
            "duration_seconds": 47.1,
            "total_records": 100,
            "success_rate": 0.99,
        })

        report = compare_run_manifests("run_a", "run_b", tmp_path, save_report=False)
        assert report["run_a"] == "run_a"
        assert report["run_b"] == "run_b"

    def test_compare_score_deltas(self, tmp_path):
        from eval_engine.comparison import compare_run_manifests

        _write_manifest(tmp_path / "run_a" / "manifest.json", {
            "experiment_id": "run_a",
            "mean_scores": {"ndcg": 0.70},
            "total_records": 50, "success_rate": 1.0,
        })
        _write_manifest(tmp_path / "run_b" / "manifest.json", {
            "experiment_id": "run_b",
            "mean_scores": {"ndcg": 0.80},
            "total_records": 50, "success_rate": 1.0,
        })

        report = compare_run_manifests("run_a", "run_b", tmp_path, save_report=False)
        assert "score_deltas" in report
        assert abs(report["score_deltas"]["ndcg"] - 0.10) < 0.001

    def test_compare_same_dataset_flag(self, tmp_path):
        from eval_engine.comparison import compare_run_manifests
        from eval_engine.dataset_manifest import DatasetManifest

        ds_manifest = DatasetManifest.from_records(SAMPLE_RECORDS)
        ds_dict = ds_manifest.to_dict()

        _write_manifest(tmp_path / "run_a" / "manifest.json", {
            "experiment_id": "run_a", "mean_scores": {}, "total_records": 10, "success_rate": 1.0
        })
        _write_manifest(tmp_path / "run_a" / "dataset_manifest.json", ds_dict)
        _write_manifest(tmp_path / "run_b" / "manifest.json", {
            "experiment_id": "run_b", "mean_scores": {}, "total_records": 10, "success_rate": 1.0
        })
        _write_manifest(tmp_path / "run_b" / "dataset_manifest.json", ds_dict)

        report = compare_run_manifests("run_a", "run_b", tmp_path, save_report=False)
        assert report.get("same_dataset") is True

    def test_compare_missing_run_returns_empty(self, tmp_path):
        from eval_engine.comparison import compare_run_manifests
        report = compare_run_manifests("nonexistent_a", "nonexistent_b", tmp_path, save_report=False)
        assert report == {}


# =============================================================================
# CLI
# =============================================================================

class TestPhase4CLI:

    def test_manifest_command_exists(self):
        from typer.testing import CliRunner
        from eval_engine.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["manifest", "--help"])
        assert result.exit_code == 0
        assert "manifest" in result.output.lower()

    def test_compare_command_exists(self):
        from typer.testing import CliRunner
        from eval_engine.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.output.lower()

    def test_run_has_seed_flag(self):
        from typer.testing import CliRunner
        from eval_engine.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert "--seed" in result.output

    def test_run_has_track_flag(self):
        from typer.testing import CliRunner
        from eval_engine.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert "--track" in result.output

    def test_oversight_run_has_seed_via_config(self):
        """oversight-run uses EvalConfig.seed (inherited from config)."""
        from eval_engine.config import EvalConfig, DatasetConfig
        config = EvalConfig(
            experiment_id="test",
            architecture="centralized",
            model="claude-haiku-4-5",
            dataset=DatasetConfig(path=Path("datasets/eval_set.json")),
            seed=123,
        )
        assert config.seed == 123
        assert config.track is False

    def test_config_track_default_false(self, tmp_path):
        """EvalConfig.track defaults to False."""
        from eval_engine.config import EvalConfig, DatasetConfig
        ds_path = tmp_path / "ds.json"
        ds_path.write_text("[]")
        config = EvalConfig(
            experiment_id="test",
            architecture="centralized",
            model="claude-haiku-4-5",
            dataset=DatasetConfig(path=ds_path),
        )
        assert config.track is False
        assert config.seed == 42

    def test_config_seed_and_track_settable(self, tmp_path):
        """EvalConfig.seed and track can be set."""
        from eval_engine.config import EvalConfig, DatasetConfig
        ds_path = tmp_path / "ds.json"
        ds_path.write_text("[]")
        config = EvalConfig(
            experiment_id="test",
            architecture="centralized",
            model="claude-haiku-4-5",
            dataset=DatasetConfig(path=ds_path),
            seed=999,
            track=True,
        )
        assert config.seed == 999
        assert config.track is True
