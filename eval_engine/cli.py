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


@app.command("list-metrics")
def list_metrics() -> None:
    """List all available metric plugins."""
    table = Table(title="Available Metrics", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Scope", style="yellow")

    scope_map = {
        "ndcg": "Retrieval Eval",
        "recall_at_k": "Consolidation Eval Suite",
        "ragas_consolidation_delta": "Consolidation Delta",
        "per_stage_ablation": "Stage Ablation",
        "threshold_compute_budget": "Compute Budget",
        "single_session_poisoning": "Poisoning Test",
        "query_perturbation": "Perturbation Test",
        "llamaguard_safety": "rag-eval-harness",
        "ragas_grounding": "rag-eval-harness",
        "session_level": "Longitudinal (reserved)",
        "longitudinal": "Longitudinal (reserved)",
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
