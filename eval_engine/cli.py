"""
eval_engine/cli.py

Verity CLI entrypoint.

Usage:
    verity run --config configs/consolidation_eval_example.yaml
    verity run --dataset datasets/eval_set.json --metrics ndcg,ragas_consolidation_delta --model claude-sonnet-4-6
    verity list-metrics
    verity validate-config configs/consolidation_eval_example.yaml

Install: pip install -e .
Then: verity --help
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from eval_engine.config import (
    AsyncConfig,
    BudgetConfig,
    DatasetConfig,
    EvalConfig,
    MetricConfig,
    RAGArchitecture,
    RetryConfig,
    StatisticsConfig,
    SupportedModel,
    TestName,
)
from eval_engine.metrics import registry

app = typer.Typer(
    name="verity",
    help="Verity Evaluation Engine — AI assurance and RAG evaluation infrastructure",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("run")
def run_eval(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML EvalConfig file. If provided, all other flags are ignored."
    ),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", "-d",
        help="Path to evaluation dataset (JSON/JSONL/CSV)"
    ),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", "-m",
        help="Comma-separated metric names, e.g. ndcg,ragas_consolidation_delta,llamaguard_safety"
    ),
    model: str = typer.Option(
        "claude-sonnet-4-6", "--model",
        help="LLM judge model string"
    ),
    architecture: str = typer.Option(
        "centralized", "--architecture", "-a",
        help="RAG architecture: centralized | decentralized | consolidation"
    ),
    experiment_id: str = typer.Option(
        "", "--experiment-id", "-e",
        help="Unique run identifier (auto-generated if not set)"
    ),
    budget: float = typer.Option(
        5.00, "--budget",
        help="Max LLM spend in USD before halt"
    ),
    concurrency: int = typer.Option(
        10, "--concurrency",
        help="Max concurrent async queries"
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for sampling, poisoning, perturbation"
    ),
    track: bool = typer.Option(
        False, "--track",
        help="Write reproducibility bundle and dataset manifest alongside results"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o",
        help="Output directory for results and cost ledger"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run an evaluation against a dataset."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load config from YAML or build from flags
    if config:
        rprint(f"[bold green]Loading config:[/bold green] {config}")
        try:
            eval_config = EvalConfig.from_yaml(config)
            # CLI flags override config file for seed/track
            if seed != 42:
                object.__setattr__(eval_config, 'seed', seed)
            if track:
                object.__setattr__(eval_config, 'track', True)
        except Exception as e:
            rprint(f"[bold red]Config validation failed:[/bold red] {e}")
            raise typer.Exit(code=1)
    else:
        if not dataset or not metrics:
            rprint("[bold red]Error:[/bold red] Either --config or both --dataset and --metrics are required.")
            raise typer.Exit(code=1)

        import time
        exp_id = experiment_id or f"run_{int(time.time())}"

        metric_list = []
        for m in metrics.split(","):
            m = m.strip()
            try:
                metric_list.append(MetricConfig(name=TestName(m)))
            except ValueError:
                rprint(f"[bold red]Unknown metric:[/bold red] '{m}'. Run 'verity list-metrics' to see available.")
                raise typer.Exit(code=1)

        try:
            arch = RAGArchitecture(architecture)
        except ValueError:
            rprint(f"[bold red]Unknown architecture:[/bold red] '{architecture}'")
            raise typer.Exit(code=1)

        try:
            eval_config = EvalConfig(
                experiment_id=exp_id,
                architecture=arch,
                model=model,
                dataset=DatasetConfig(path=dataset),
                metrics=metric_list,
                budget=BudgetConfig(max_usd=budget),
                async_cfg=AsyncConfig(max_concurrent_queries=concurrency),
                output_dir=output_dir,
                seed=seed,
                track=track,
            )
        except Exception as e:
            rprint(f"[bold red]Config error:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Print run summary
    table = Table(title=f"Eval Run: {eval_config.experiment_id}", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Architecture", eval_config.architecture.value)
    table.add_row("Model", str(eval_config.model))
    table.add_row("Dataset", str(eval_config.dataset.path))
    table.add_row("Metrics", ", ".join(m.name.value for m in eval_config.enabled_metrics))
    table.add_row("Budget", f"${eval_config.budget.max_usd:.2f}")
    table.add_row("Seed", str(eval_config.seed))
    table.add_row("Track", str(eval_config.track))
    table.add_row("Concurrency", str(eval_config.async_cfg.max_concurrent_queries))
    table.add_row("Output Dir", str(eval_config.output_dir))
    console.print(table)

    # Load dataset
    ds_path = eval_config.dataset.path
    if not ds_path.exists():
        rprint(f"[bold red]Dataset not found:[/bold red] {ds_path}")
        raise typer.Exit(code=1)

    dataset_records = _load_dataset(eval_config)

    # Run
    from eval_engine.runner import EvalRunner
    runner = EvalRunner(eval_config)

    try:
        result = asyncio.run(runner.run(dataset_records))
    except Exception as e:
        rprint(f"[bold red]Run failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Print results table
    results_table = Table(title="Results Summary", show_header=True)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Mean Score", style="green")
    results_table.add_column("Records", style="white")

    for m in eval_config.enabled_metrics:
        mean = result.mean_score(m.name.value)
        score_str = f"{mean:.4f}" if mean is not None else "N/A"
        results_table.add_row(m.name.value, score_str, str(len(result.records)))

    console.print(results_table)
    rprint(f"\n[bold green]Results saved to:[/bold green] {eval_config.output_dir / eval_config.experiment_id}/")


@app.command("compare")
def compare_runs(
    run_a: str = typer.Option(..., "--run-a", "-a", help="Experiment ID or path of run A"),
    run_b: str = typer.Option(..., "--run-b", "-b", help="Experiment ID or path of run B"),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o"),
) -> None:
    """
    Compare two eval runs side by side.

    Shows: config diffs, metric score deltas, cost deltas, verdict changes.

    Examples:
        verity compare --run-a consolidation_eval_run1 --run-b consolidation_eval_run2
        verity compare --run-a outputs/run1 --run-b outputs/run2
    """
    from eval_engine.comparison import compare_run_manifests
    compare_run_manifests(run_a, run_b, output_dir)


@app.command("manifest")
def dataset_manifest_cmd(
    dataset: Path = typer.Argument(..., help="Path to dataset file (JSON/JSONL/CSV)"),
    version_tag: Optional[str] = typer.Option(
        None, "--version-tag", "-v",
        help="Human-readable version label, e.g. 'v1.0', 'poisoned-10pct'"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Save manifest to this path (default: prints to console only)"
    ),
) -> None:
    """
    Generate a dataset manifest (content hash + schema) without running an eval.

    Use this to version and identify datasets before running experiments.

    Examples:
        verity manifest datasets/eval_set.json
        verity manifest datasets/eval_set.json --version-tag v1.0 --output manifests/
    """
    from eval_engine.dataset_manifest import DatasetManifest

    if not dataset.exists():
        rprint(f"[bold red]Dataset not found:[/bold red] {dataset}")
        raise typer.Exit(code=1)

    try:
        manifest = DatasetManifest.from_file(dataset, version_tag=version_tag)
        manifest.print_summary()

        if output:
            saved = manifest.save(output)
            rprint(f"[bold green]Manifest saved:[/bold green] {saved}")
        else:
            rprint("[dim]Use --output to save manifest to a file.[/dim]")

    except Exception as e:
        rprint(f"[bold red]Manifest generation failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("oversight-run")
def oversight_run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML EvalConfig file."
    ),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", "-d",
        help="Path to evaluation dataset (JSON/JSONL/CSV)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Run without real API calls (architecture testing, zero cost)"
    ),
    proposer_model: str = typer.Option(
        "claude-haiku-4-5", "--proposer-model",
        help="Model for Agent A (Proposer)"
    ),
    critic_model: str = typer.Option(
        "claude-sonnet-4-6", "--critic-model",
        help="Model for Agent B (Critic) — reward hacking detection"
    ),
    judge_model: str = typer.Option(
        "claude-haiku-4-5", "--judge-model",
        help="Model for Agent C (Judge)"
    ),
    recall_at_k: int = typer.Option(
        5, "--recall-at-k",
        help="Number of context chunks critic verifies against"
    ),
    budget: float = typer.Option(
        10.00, "--budget",
        help="Max LLM spend in USD before halt"
    ),
    concurrency: int = typer.Option(
        4, "--concurrency",
        help="Max concurrent debate rounds"
    ),
    use_celery: bool = typer.Option(
        False, "--celery",
        help="Dispatch to Celery workers (requires Redis running)"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Run dataset through the multi-agent oversight pipeline (Proposer → Critic → Judge).

    Each record is evaluated for safety, accuracy, and reward hacking.
    Results include per-debate traces, cost ledger, and statistical summary.

    Examples:
        verity oversight-run --config configs/consolidation_eval_example.yaml --dry-run
        verity oversight-run --dataset datasets/eval_set.json --budget 20.00
        verity oversight-run --dataset datasets/eval_set.json --celery
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load or build config
    if config:
        try:
            eval_config = EvalConfig.from_yaml(config)
        except Exception as e:
            rprint(f"[bold red]Config error:[/bold red] {e}")
            raise typer.Exit(code=1)
    else:
        if not dataset:
            rprint("[bold red]Error:[/bold red] Either --config or --dataset is required.")
            raise typer.Exit(code=1)

        import time
        eval_config = EvalConfig(
            experiment_id=f"oversight_{int(time.time())}",
            architecture="centralized",
            model=critic_model,
            dataset=DatasetConfig(path=dataset),
            metrics=[],
            budget=BudgetConfig(max_usd=budget),
            async_cfg=AsyncConfig(max_concurrent_queries=concurrency),
            output_dir=output_dir,
        )

    # Print run summary
    table = Table(title=f"Oversight Run: {eval_config.experiment_id}", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Proposer", proposer_model)
    table.add_row("Critic", critic_model)
    table.add_row("Judge", judge_model)
    table.add_row("Dataset", str(eval_config.dataset.path))
    table.add_row("Dry Run", str(dry_run))
    table.add_row("Recall@K", str(recall_at_k))
    table.add_row("Budget", f"${eval_config.budget.max_usd:.2f}")
    table.add_row("Concurrency", str(eval_config.async_cfg.max_concurrent_queries))
    table.add_row("Celery", str(use_celery))
    table.add_row("Output Dir", str(eval_config.output_dir))
    console.print(table)

    if dry_run:
        rprint("[bold yellow]DRY-RUN MODE — no API calls will be made[/bold yellow]")

    # Load dataset
    if not eval_config.dataset.path.exists():
        rprint(f"[bold red]Dataset not found:[/bold red] {eval_config.dataset.path}")
        raise typer.Exit(code=1)

    dataset_records = _load_dataset(eval_config)
    rprint(f"Loaded [bold]{len(dataset_records)}[/bold] records")

    # Run oversight pipeline
    from eval_engine.orchestration.oversight_runner import OversightRunner

    runner = OversightRunner(
        config=eval_config,
        dry_run=dry_run,
        proposer_model=proposer_model,
        critic_model=critic_model,
        judge_model=judge_model,
        use_celery=use_celery,
        recall_at_k=recall_at_k,
    )

    try:
        result = asyncio.run(runner.run(dataset_records))
    except Exception as e:
        rprint(f"[bold red]Oversight run failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Print results table
    results_table = Table(title="Oversight Results Summary", show_header=True)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Total Debates", str(len(result.debate_results)))
    results_table.add_row("Success Rate", f"{result.success_rate:.1%}")
    results_table.add_row("Reward Hacking Rate", f"{result.reward_hacking_rate:.1%}")
    results_table.add_row("Mean Safety Score", f"{result.mean_safety_score():.4f}" if result.mean_safety_score() else "N/A")
    results_table.add_row("Mean Accuracy Score", f"{result.mean_accuracy_score():.4f}" if result.mean_accuracy_score() else "N/A")

    for verdict, count in result.verdict_distribution.items():
        results_table.add_row(f"Verdict: {verdict}", str(count))

    console.print(results_table)
    rprint(f"\n[bold green]Results saved to:[/bold green] {eval_config.output_dir / eval_config.experiment_id}/")


@app.command("list-metrics")
def list_metrics() -> None:
    """List all available metric plugins."""
    table = Table(title="Available Metrics", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="yellow")

    scope_map = {
        # Implemented
        "ndcg":                      "Retrieval",
        "recall_at_k":               "Retrieval",
        "mean_reciprocal_rank":      "Retrieval",
        "ragas_grounding":           "Grounding",
        "ragas_consolidation_delta": "Consolidation",
        "compression_delta":         "Graph",
        "deduplication_delta":       "Graph",
        "entity_coverage":           "Graph",
        "llamaguard_safety":         "Safety",
        "per_stage_ablation":        "Ablation",
        "threshold_compute_budget":  "Compute",
        "single_session_poisoning":  "Adversarial",
        "query_perturbation":        "Adversarial",
        "calibration":               "Trust",
        "hallucination_rate":        "Trust",
        "trust_score":               "Trust",
        "multi_session_persistence": "Longitudinal",
        # Tier 2 scaffolds
        "consistency":               "Scaffold (Tier 2)",
        "constitutional_eval":       "Scaffold (Tier 2)",
        "model_written_eval":        "Scaffold (Tier 2)",
        "source_reliability":        "Scaffold (Tier 2)",
        # Tier 3 scaffolds
        "goal_misgeneralization":    "Scaffold (Tier 3)",
        "deceptive_alignment":       "Scaffold (Tier 3)",
    }

    for name in registry.available:
        scope = scope_map.get(name, "—")
        table.add_row(name, scope)

    console.print(table)


@app.command("validate-config")
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to YAML config file")
) -> None:
    """Validate an EvalConfig YAML file without running."""
    try:
        cfg = EvalConfig.from_yaml(config_path)
        rprint(f"[bold green]✓ Config valid:[/bold green] {config_path}")
        rprint(f"  Experiment ID:  {cfg.experiment_id}")
        rprint(f"  Architecture:   {cfg.architecture.value}")
        rprint(f"  Model:          {cfg.model}")
        rprint(f"  Metrics:        {[m.name.value for m in cfg.enabled_metrics]}")
        rprint(f"  Budget:         ${cfg.budget.max_usd:.2f}")
    except Exception as e:
        rprint(f"[bold red]✗ Config invalid:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset(config: EvalConfig) -> list[dict]:
    path = config.dataset.path
    fmt = config.dataset.format

    if fmt in ("json",):
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    elif fmt == "jsonl":
        import jsonlines
        with jsonlines.open(path) as reader:
            return list(reader)

    elif fmt == "csv":
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")

    elif fmt == "parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
