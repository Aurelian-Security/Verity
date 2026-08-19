"""
eval_engine/orchestration/oversight_runner.py

OversightRunner — Batch Debate Pipeline Executor

PURPOSE:
    Wraps DebateRound the same way EvalRunner wraps metric evaluation.
    Takes a dataset, runs every record through the A→B→C debate pipeline,
    collects DebateResult objects, feeds them into StatEngine, and writes
    a structured output manifest.

    This is the missing wiring layer between:
        DebateRound (single query)  ←→  OversightRunner (full dataset)
        EvalRunner (metric evals)   ←→  OversightRunner (oversight evals)

    Both runners produce compatible output manifests and cost ledgers,
    so results can be compared and combined in Phase 4 reporting.

INTEGRATION:
    - Reads EvalConfig for model, budget, async, and output settings
    - Uses CostTracker for per-agent token and cost accounting
    - Feeds DebateResult batches into StatEngine for significance testing
    - Writes per-query debate JSONs + batch manifest + cost ledger
    - Dispatches via Celery (distributed) or sync fallback

OUTPUTS per run:
    outputs/{experiment_id}/
        oversight_results.jsonl     ← per-query DebateResult (streamed)
        oversight_manifest.json     ← batch summary + stats
        cost_ledger_{id}.jsonl      ← per-agent token/cost log
        debates/
            debate_{query_id}.json  ← full trace per query
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_engine.config import EvalConfig
from eval_engine.cost_tracker import BudgetExceededError, CostTracker
from eval_engine.orchestration.debate_round import DebateResult, DebateRound
from eval_engine.orchestration.celery_tasks import dispatch_debate_batch, collect_debate_results
from eval_engine.statistics import StatEngine, StatReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oversight run result
# ---------------------------------------------------------------------------

@dataclass
class OversightRunResult:
    """
    Aggregated results from a full oversight pipeline run.
    Analogous to RunResult from runner.py.
    """
    experiment_id: str
    debate_results: list[DebateResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    dry_run: bool = False

    def add_result(self, result: DebateResult) -> None:
        self.debate_results.append(result)

    def add_error(self, query_id: str, error: str) -> None:
        self.errors.append({"query_id": query_id, "error": error})

    def finalize(self) -> None:
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    @property
    def success_rate(self) -> float:
        total = len(self.debate_results) + len(self.errors)
        return len(self.debate_results) / total if total > 0 else 0.0

    @property
    def reward_hacking_rate(self) -> float:
        """Fraction of debates where reward hacking was confirmed."""
        if not self.debate_results:
            return 0.0
        confirmed = sum(1 for r in self.debate_results if r.reward_hacking_confirmed)
        return confirmed / len(self.debate_results)

    @property
    def verdict_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {"Pass": 0, "Conditional": 0, "Fail": 0}
        for r in self.debate_results:
            dist[r.verdict] = dist.get(r.verdict, 0) + 1
        return dist

    def mean_safety_score(self) -> float | None:
        scores = [r.final_safety_score for r in self.debate_results]
        return sum(scores) / len(scores) if scores else None

    def mean_accuracy_score(self) -> float | None:
        scores = [r.final_accuracy_score for r in self.debate_results]
        return sum(scores) / len(scores) if scores else None

    def safety_scores(self) -> list[float]:
        return [r.final_safety_score for r in self.debate_results]

    def accuracy_scores(self) -> list[float]:
        return [r.final_accuracy_score for r in self.debate_results]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "pipeline": "oversight",
            "dry_run": self.dry_run,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_debates": len(self.debate_results),
            "total_errors": len(self.errors),
            "success_rate": round(self.success_rate, 4),
            "reward_hacking_rate": round(self.reward_hacking_rate, 4),
            "verdict_distribution": self.verdict_distribution,
            "mean_safety_score": self.mean_safety_score(),
            "mean_accuracy_score": self.mean_accuracy_score(),
        }


# ---------------------------------------------------------------------------
# OversightRunner
# ---------------------------------------------------------------------------

class OversightRunner:
    """
    Batch oversight pipeline executor.

    Runs a full dataset through the A→B→C debate pipeline,
    integrates cost tracking, statistical analysis, and structured output.

    Usage:
        config = EvalConfig.from_yaml("configs/consolidation_eval_example.yaml")
        runner = OversightRunner(config, dry_run=False)
        result = await runner.run(dataset)
    """

    def __init__(
        self,
        config: EvalConfig,
        dry_run: bool = False,
        proposer_model: str | None = None,
        critic_model: str | None = None,
        judge_model: str | None = None,
        use_celery: bool = False,
        recall_at_k: int = 5,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.proposer_model = proposer_model
        self.critic_model = critic_model
        self.judge_model = judge_model
        self.use_celery = use_celery
        self.recall_at_k = recall_at_k
        self.cost_tracker = CostTracker(
            budget_usd=config.budget.max_usd,
            warn_at_pct=config.budget.warn_at_pct,
        )
        self.stat_engine = StatEngine(alpha=config.statistics.alpha)

    async def run(
        self,
        dataset: list[dict[str, Any]],
    ) -> OversightRunResult:
        """
        Run full dataset through oversight pipeline.

        Each record must have: query_id (optional), question, contexts (list[str]).
        Optional: answer (used as proposer context hint, not required).

        Returns OversightRunResult with all DebateResults and stats.
        """
        cfg = self.config
        run_result = OversightRunResult(
            experiment_id=cfg.experiment_id,
            dry_run=self.dry_run,
        )

        # Sample if configured
        if cfg.dataset.sample_n and cfg.dataset.sample_n < len(dataset):
            import random
            dataset = random.sample(dataset, cfg.dataset.sample_n)
            logger.info(f"Sampled {cfg.dataset.sample_n} records")

        # Set up output dirs
        output_dir = cfg.output_dir / cfg.experiment_id
        debates_dir = output_dir / "debates"
        output_dir.mkdir(parents=True, exist_ok=True)
        debates_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "oversight_results.jsonl"
        manifest_path = output_dir / "oversight_manifest.json"

        logger.info(
            f"OversightRunner starting: {cfg.experiment_id} | "
            f"{len(dataset)} records | dry_run={self.dry_run}"
        )

        # Normalize records
        records = [self._normalize_record(r, i) for i, r in enumerate(dataset)]

        if self.use_celery:
            # Distributed path
            debate_results = await self._run_celery(records, debates_dir)
        else:
            # Async local path
            debate_results = await self._run_async(records, debates_dir)

        # Collect results
        with open(results_path, "w") as f:
            for result in debate_results:
                if isinstance(result, DebateResult):
                    run_result.add_result(result)
                    row = {
                        "query_id": result.query_id,
                        "verdict": result.verdict,
                        "safety": result.final_safety_score,
                        "accuracy": result.final_accuracy_score,
                        "reward_hacking": result.reward_hacking_confirmed,
                        "latency": result.total_latency_seconds,
                    }
                    f.write(json.dumps(row) + "\n")

                    # Feed into cost tracker
                    if result.total_prompt_tokens > 0:
                        self.cost_tracker.record(
                            model=str(cfg.model),
                            metric_name="oversight_debate",
                            prompt_tokens=result.total_prompt_tokens,
                            completion_tokens=result.total_completion_tokens,
                            latency_seconds=result.total_latency_seconds,
                        )
                elif isinstance(result, dict) and "error" in result:
                    run_result.add_error(
                        result.get("query_id", "unknown"),
                        result.get("error", "unknown error")
                    )

        run_result.finalize()

        # Statistical analysis on scores
        stat_report = self._run_statistics(run_result, output_dir)

        # Write manifest
        manifest = run_result.to_manifest()
        manifest["statistics"] = stat_report.to_dict() if stat_report else {}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")

        # Save cost ledger
        if cfg.budget.track_tokens:
            self.cost_tracker.save_ledger(output_dir, cfg.experiment_id)

        self._log_summary(run_result)
        return run_result

    async def _run_async(
        self,
        records: list[dict[str, Any]],
        debates_dir: Path,
    ) -> list[DebateResult | dict]:
        """Run debates asynchronously with semaphore concurrency control."""
        semaphore = asyncio.Semaphore(self.config.async_cfg.max_concurrent_queries)
        results = []

        async def run_one(record: dict) -> DebateResult | dict:
            async with semaphore:
                try:
                    # Check budget before each debate
                    self.cost_tracker.check_budget()

                    round_runner = DebateRound(
                        dry_run=self.dry_run,
                        proposer_model=self.proposer_model,
                        critic_model=self.critic_model,
                        judge_model=self.judge_model,
                    )

                    # Run in executor (DebateRound is synchronous)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda r=record: round_runner.run(
                            query_id=r["query_id"],
                            query=r["query"],
                            contexts=r.get("contexts", []),
                            recall_at_k=self.recall_at_k,
                        )
                    )

                    # Save individual debate trace
                    result.save(debates_dir, f"debate_{result.query_id}.json")
                    return result

                except BudgetExceededError as e:
                    logger.error(f"Budget exceeded: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Debate failed for {record.get('query_id')}: {e}")
                    return {"query_id": record.get("query_id"), "error": str(e)}

        # Process in batches
        batch_size = self.config.async_cfg.batch_size
        batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]

        for b_idx, batch in enumerate(batches):
            logger.info(f"Oversight batch {b_idx+1}/{len(batches)} ({len(batch)} debates)")
            try:
                tasks = [run_one(r) for r in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, BudgetExceededError):
                        raise r
                    results.append(r)
            except BudgetExceededError:
                logger.error("Budget exceeded — halting oversight run.")
                break

        return results

    async def _run_celery(
        self,
        records: list[dict[str, Any]],
        debates_dir: Path,
    ) -> list[DebateResult | dict]:
        """Dispatch to Celery workers and collect results."""
        logger.info(f"Dispatching {len(records)} debates to Celery workers")

        handles = dispatch_debate_batch(
            records=records,
            dry_run=self.dry_run,
            proposer_model=self.proposer_model,
            critic_model=self.critic_model,
            judge_model=self.judge_model,
            recall_at_k=self.recall_at_k,
            output_dir=str(debates_dir),
            use_celery=True,
        )

        raw_results = collect_debate_results(
            handles,
            timeout_seconds=self.config.async_cfg.timeout_seconds * len(records),
        )

        # Convert dicts back to DebateResult-like objects for uniform handling
        return raw_results

    def _normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        """Ensure each record has query_id, query, contexts."""
        return {
            "query_id": record.get("query_id", f"q_{idx:04d}"),
            "query": record.get(
                self.config.dataset.question_col,
                record.get("query", record.get("question", ""))
            ),
            "contexts": record.get(
                self.config.dataset.context_col,
                record.get("contexts", [])
            ),
        }

    def _run_statistics(
        self,
        result: OversightRunResult,
        output_dir: Path,
    ) -> StatReport | None:
        """Run statistical analysis on safety and accuracy score distributions."""
        safety = result.safety_scores()
        accuracy = result.accuracy_scores()

        if len(safety) < 3:
            logger.warning(
                f"[OversightRunner] Only {len(safety)} results — "
                f"skipping statistical analysis (minimum 3 required)."
            )
            return None

        all_results = []

        # Descriptive stats for safety and accuracy
        all_results.append(
            self.stat_engine.descriptive(safety, label="safety_score_descriptive")
        )
        all_results.append(
            self.stat_engine.descriptive(accuracy, label="accuracy_score_descriptive")
        )

        # Normality checks
        all_results.append(
            self.stat_engine.shapiro(safety, label="safety_normality")
        )
        all_results.append(
            self.stat_engine.shapiro(accuracy, label="accuracy_normality")
        )

        # Correlation between safety and accuracy
        if len(safety) >= 3:
            all_results.append(
                self.stat_engine.pearson(safety, accuracy, label="safety_accuracy_correlation")
            )

        report = StatReport(
            all_results,
            experiment_id=result.experiment_id,
            notes="Oversight pipeline statistical analysis",
        )

        stats_path = output_dir / "oversight_stats.json"
        report.save(stats_path)
        report.print_summary()

        return report

    def _log_summary(self, result: OversightRunResult) -> None:
        logger.info(f"\n{'='*55}")
        logger.info(f"Oversight run complete: {result.experiment_id}")
        logger.info(f"Duration: {result.duration_seconds:.1f}s")
        logger.info(f"Debates: {len(result.debate_results)} success, {len(result.errors)} errors")
        logger.info(f"Success rate: {result.success_rate:.1%}")
        logger.info(f"Reward hacking rate: {result.reward_hacking_rate:.1%}")
        logger.info(f"Verdict distribution: {result.verdict_distribution}")
        logger.info(f"Mean safety score: {result.mean_safety_score():.4f}" if result.mean_safety_score() else "Mean safety: N/A")
        logger.info(f"Mean accuracy score: {result.mean_accuracy_score():.4f}" if result.mean_accuracy_score() else "Mean accuracy: N/A")
        self.cost_tracker.print_summary()
