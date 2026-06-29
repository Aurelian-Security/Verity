"""
eval_engine/comparison.py

Run Comparison Engine — verity compare

PURPOSE:
    Diffs two eval run manifests side by side.
    Shows: config changes, metric score deltas, cost deltas,
    verdict distribution changes, dataset identity verification.

    Used during Paper 1 revision to compare:
        - Pre vs post consolidation runs
        - Centralized vs decentralized architecture runs
        - Baseline vs poisoned corpus runs
        - Different model configurations

USAGE:
    # CLI
    verity compare --run-a run_001 --run-b run_002

    # Programmatic
    from eval_engine.comparison import compare_run_manifests
    report = compare_run_manifests("run_001", "run_002", output_dir)

OUTPUT:
    Prints comparison table to console.
    Saves comparison_report.json to outputs/ if runs have reproducibility bundles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _load_manifest(run_id: str, output_dir: Path) -> dict[str, Any]:
    """
    Load manifest.json or oversight_manifest.json for a run.
    Accepts run ID (looks in output_dir/run_id/) or direct path.
    """
    # Try as direct path first
    direct = Path(run_id)
    if direct.is_dir():
        for fname in ("manifest.json", "oversight_manifest.json"):
            p = direct / fname
            if p.exists():
                with open(p) as f:
                    return json.load(f)

    # Try as experiment_id under output_dir
    run_dir = output_dir / run_id
    if run_dir.is_dir():
        for fname in ("manifest.json", "oversight_manifest.json"):
            p = run_dir / fname
            if p.exists():
                with open(p) as f:
                    return json.load(f)

    raise FileNotFoundError(
        f"Could not find manifest for run '{run_id}'. "
        f"Looked in: {direct}, {run_dir}. "
        f"Make sure the experiment_id matches an existing outputs/ subdirectory."
    )


def _load_reproducibility(run_id: str, output_dir: Path) -> dict[str, Any] | None:
    """Load reproducibility.json for a run if it exists."""
    for base in (Path(run_id), output_dir / run_id):
        p = base / "reproducibility.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _load_dataset_manifest(run_id: str, output_dir: Path) -> dict[str, Any] | None:
    """Load dataset_manifest.json for a run if it exists."""
    for base in (Path(run_id), output_dir / run_id):
        p = base / "dataset_manifest.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _delta(a: float | None, b: float | None) -> str:
    """Format delta between two float values."""
    if a is None or b is None:
        return "N/A"
    diff = b - a
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


def _pct_change(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "N/A"
    pct = ((b - a) / abs(a)) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def compare_run_manifests(
    run_a_id: str,
    run_b_id: str,
    output_dir: Path | str = Path("outputs"),
    save_report: bool = True,
) -> dict[str, Any]:
    """
    Compare two eval run manifests and print a side-by-side diff.

    Args:
        run_a_id:   Experiment ID or path of baseline run (Run A)
        run_b_id:   Experiment ID or path of comparison run (Run B)
        output_dir: Root output directory (default: outputs/)
        save_report: Save comparison_report.json to output_dir

    Returns:
        Comparison report dict
    """
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console()
    output_dir = Path(output_dir)

    # Load manifests
    try:
        manifest_a = _load_manifest(run_a_id, output_dir)
    except FileNotFoundError as e:
        rprint(f"[bold red]Run A not found:[/bold red] {e}")
        return {}

    try:
        manifest_b = _load_manifest(run_b_id, output_dir)
    except FileNotFoundError as e:
        rprint(f"[bold red]Run B not found:[/bold red] {e}")
        return {}

    repro_a = _load_reproducibility(run_a_id, output_dir)
    repro_b = _load_reproducibility(run_b_id, output_dir)
    ds_a = _load_dataset_manifest(run_a_id, output_dir)
    ds_b = _load_dataset_manifest(run_b_id, output_dir)

    report: dict[str, Any] = {
        "run_a": run_a_id,
        "run_b": run_b_id,
    }

    # -----------------------------------------------------------------------
    # Run identity
    # -----------------------------------------------------------------------
    id_table = Table(title="Run Identity", show_header=True)
    id_table.add_column("Field", style="cyan")
    id_table.add_column("Run A", style="white")
    id_table.add_column("Run B", style="white")

    id_table.add_row(
        "Experiment ID",
        manifest_a.get("experiment_id", "?"),
        manifest_b.get("experiment_id", "?"),
    )

    if repro_a and repro_b:
        id_table.add_row(
            "Git Hash",
            (repro_a.get("git", {}).get("hash") or "N/A")[:12],
            (repro_b.get("git", {}).get("hash") or "N/A")[:12],
        )
        id_table.add_row(
            "Seed",
            str(repro_a.get("seed", "N/A")),
            str(repro_b.get("seed", "N/A")),
        )
        id_table.add_row(
            "Verity Version",
            repro_a.get("verity_version", "N/A"),
            repro_b.get("verity_version", "N/A"),
        )
        report["seeds_match"] = repro_a.get("seed") == repro_b.get("seed")
        report["git_hashes"] = {
            "a": repro_a.get("git", {}).get("hash"),
            "b": repro_b.get("git", {}).get("hash"),
        }

    console.print(id_table)

    # -----------------------------------------------------------------------
    # Dataset identity
    # -----------------------------------------------------------------------
    if ds_a or ds_b:
        ds_table = Table(title="Dataset Identity", show_header=True)
        ds_table.add_column("Field", style="cyan")
        ds_table.add_column("Run A", style="white")
        ds_table.add_column("Run B", style="white")

        ds_id_a = (ds_a or {}).get("dataset_id", "N/A")[:20] + "..."
        ds_id_b = (ds_b or {}).get("dataset_id", "N/A")[:20] + "..."
        same_dataset = (ds_a or {}).get("dataset_id") == (ds_b or {}).get("dataset_id")

        ds_table.add_row("Dataset ID", ds_id_a, ds_id_b)
        ds_table.add_row("Same Dataset", str(same_dataset), "")
        ds_table.add_row(
            "Row Count",
            str((ds_a or {}).get("row_count", "N/A")),
            str((ds_b or {}).get("row_count", "N/A")),
        )
        ds_table.add_row(
            "Version Tag",
            str((ds_a or {}).get("version_tag") or "untagged"),
            str((ds_b or {}).get("version_tag") or "untagged"),
        )

        console.print(ds_table)
        report["same_dataset"] = same_dataset

        if not same_dataset:
            rprint("[bold yellow]⚠  WARNING: Runs used different datasets — comparison may not be valid.[/bold yellow]")

    # -----------------------------------------------------------------------
    # Metric score deltas (eval runs)
    # -----------------------------------------------------------------------
    scores_a = manifest_a.get("mean_scores", {})
    scores_b = manifest_b.get("mean_scores", {})

    if scores_a or scores_b:
        all_metrics = sorted(set(list(scores_a.keys()) + list(scores_b.keys())))
        score_table = Table(title="Metric Score Comparison", show_header=True)
        score_table.add_column("Metric", style="cyan")
        score_table.add_column("Run A", style="white")
        score_table.add_column("Run B", style="white")
        score_table.add_column("Delta (B-A)", style="green")
        score_table.add_column("% Change", style="yellow")

        score_deltas = {}
        for metric in all_metrics:
            a_val = scores_a.get(metric)
            b_val = scores_b.get(metric)
            delta_str = _delta(a_val, b_val)
            pct_str = _pct_change(a_val, b_val)

            score_table.add_row(
                metric,
                f"{a_val:.4f}" if a_val is not None else "N/A",
                f"{b_val:.4f}" if b_val is not None else "N/A",
                delta_str,
                pct_str,
            )
            if a_val is not None and b_val is not None:
                score_deltas[metric] = round(b_val - a_val, 6)

        console.print(score_table)
        report["score_deltas"] = score_deltas

    # -----------------------------------------------------------------------
    # Oversight verdict comparison
    # -----------------------------------------------------------------------
    verd_a = manifest_a.get("verdict_distribution", {})
    verd_b = manifest_b.get("verdict_distribution", {})

    if verd_a or verd_b:
        verd_table = Table(title="Verdict Distribution", show_header=True)
        verd_table.add_column("Verdict", style="cyan")
        verd_table.add_column("Run A", style="white")
        verd_table.add_column("Run B", style="white")
        verd_table.add_column("Delta", style="green")

        for verdict in ["Pass", "Conditional", "Fail"]:
            a_count = verd_a.get(verdict, 0)
            b_count = verd_b.get(verdict, 0)
            delta = b_count - a_count
            sign = "+" if delta >= 0 else ""
            verd_table.add_row(verdict, str(a_count), str(b_count), f"{sign}{delta}")

        rh_a = manifest_a.get("reward_hacking_rate")
        rh_b = manifest_b.get("reward_hacking_rate")
        if rh_a is not None and rh_b is not None:
            verd_table.add_row(
                "RH Rate",
                f"{rh_a:.1%}",
                f"{rh_b:.1%}",
                _delta(rh_a, rh_b),
            )

        console.print(verd_table)

    # -----------------------------------------------------------------------
    # Cost comparison
    # -----------------------------------------------------------------------
    cost_table = Table(title="Run Statistics", show_header=True)
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Run A", style="white")
    cost_table.add_column("Run B", style="white")

    for field, label in [
        ("duration_seconds", "Duration (s)"),
        ("total_records", "Records"),
        ("success_rate", "Success Rate"),
    ]:
        a_val = manifest_a.get(field)
        b_val = manifest_b.get(field)
        cost_table.add_row(
            label,
            str(round(a_val, 3)) if isinstance(a_val, float) else str(a_val or "N/A"),
            str(round(b_val, 3)) if isinstance(b_val, float) else str(b_val or "N/A"),
        )

    console.print(cost_table)

    # Save report
    if save_report:
        report_path = output_dir / f"compare_{run_a_id}_vs_{run_b_id}.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            rprint(f"\n[bold green]Comparison report saved:[/bold green] {report_path}")
        except Exception as e:
            logger.warning(f"Could not save comparison report: {e}")

    return report
