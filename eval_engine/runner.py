"""
eval_engine/runner.py — Async EvalRunner with sanitizer integration.
FIX-7: InputSanitizer applied before every judge call.
Output: results.jsonl (streamed) + manifest.json (run summary).
"""
from __future__ import annotations
import asyncio, json, logging, time, uuid
from pathlib import Path
from typing import Any

from eval_engine.config import EvalConfig, metric_name_value
from eval_engine.cost_tracker import BudgetExceededError, CostTracker
from eval_engine.metrics import MetricResult, MetricsRegistry, registry as default_registry
from eval_engine.sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class RunResult:
    def __init__(self, experiment_id: str, config: EvalConfig) -> None:
        self.experiment_id = experiment_id
        self.config = config
        self.records: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.start_time: float = time.time()
        self.end_time: float | None = None
        self.sanitizer_summary: dict[str, int] = {}

    def add_record(self, query_idx: int, metric_results: list[MetricResult], query: str) -> None:
        self.records.append({
            "query_idx": query_idx, "query": query,
            "results": {r.metric_name: r.score for r in metric_results},
            "raw": {r.metric_name: r.raw for r in metric_results},
            "errors": {r.metric_name: r.error for r in metric_results if not r.succeeded},
        })

    def add_error(self, query_idx: int, error: str) -> None:
        self.errors.append({"query_idx": query_idx, "error": error})

    def finalize(self, sanitizer: InputSanitizer | None = None) -> None:
        self.end_time = time.time()
        if sanitizer:
            self.sanitizer_summary = sanitizer.detection_summary()

    @property
    def duration_seconds(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    @property
    def success_rate(self) -> float:
        total = len(self.records) + len(self.errors)
        return len(self.records) / total if total > 0 else 0.0

    def mean_score(self, metric_name: str) -> float | None:
        scores = [r["results"].get(metric_name) for r in self.records if r["results"].get(metric_name) is not None]
        return sum(scores) / len(scores) if scores else None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_records": len(self.records),
            "total_errors": len(self.errors),
            "success_rate": round(self.success_rate, 4),
            "sanitizer_detections": self.sanitizer_summary,
            "mean_scores": {
                metric_name_value(m.name): self.mean_score(metric_name_value(m.name))
                for m in self.config.enabled_metrics
            },
        }


class EvalRunner:
    def __init__(self, config: EvalConfig, metrics_registry: MetricsRegistry | None = None) -> None:
        self.config = config
        self.registry = metrics_registry or default_registry
        self.cost_tracker = CostTracker(budget_usd=config.budget.max_usd, warn_at_pct=config.budget.warn_at_pct)
        self.sanitizer = InputSanitizer(strict=False)  # FIX-7
        self._semaphore: asyncio.Semaphore | None = None

    async def run(self, dataset: list[dict[str, Any]]) -> RunResult:
        cfg = self.config
        run_result = RunResult(cfg.experiment_id, cfg)

        # Phase 4: Set global seed for reproducibility
        from eval_engine.reproducibility import set_global_seed
        set_global_seed(cfg.seed)

        if cfg.dataset.sample_n and cfg.dataset.sample_n < len(dataset):
            import random
            random.seed(cfg.seed)
            dataset = random.sample(dataset, cfg.dataset.sample_n)

        output_dir = cfg.output_dir / cfg.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"
        manifest_path = output_dir / "manifest.json"

        # Phase 4: Capture reproducibility bundle if --track
        if cfg.track:
            from eval_engine.reproducibility import ReproducibilityBundle
            from eval_engine.dataset_manifest import DatasetManifest
            repro = ReproducibilityBundle.from_config(cfg, seed=cfg.seed)
            repro.capture()
            repro.save(output_dir)
            repro.print_summary()
            if cfg.dataset.path.exists():
                ds_manifest = DatasetManifest.from_file(cfg.dataset.path)
                ds_manifest.save(output_dir)
                ds_manifest.print_summary()

        metrics = []
        for metric_cfg in cfg.enabled_metrics:
            try:
                metrics.append((metric_cfg, self.registry.from_config(metric_cfg)))
            except Exception as e:
                logger.error(f"Failed to init metric '{metric_cfg.name}': {e}")

        if not metrics:
            raise RuntimeError("No metrics initialized. Check EvalConfig.")

        logger.info(f"Run: {cfg.experiment_id} | {len(dataset)} records | {len(metrics)} metrics | seed={cfg.seed}")
        self._semaphore = asyncio.Semaphore(cfg.async_cfg.max_concurrent_queries)
        batch_size = cfg.async_cfg.batch_size
        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

        with open(results_path, "w") as f:
            for b_idx, batch in enumerate(batches):
                logger.info(f"Batch {b_idx+1}/{len(batches)}")
                tasks = [self._eval_record(b_idx * batch_size + i, rec, metrics) for i, rec in enumerate(batch)]
                for q_idx, outcome in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
                    g_idx = b_idx * batch_size + q_idx
                    if isinstance(outcome, BudgetExceededError):
                        run_result.finalize(self.sanitizer)
                        raise outcome
                    if isinstance(outcome, Exception):
                        run_result.add_error(g_idx, str(outcome))
                        continue
                    q, results = outcome
                    run_result.add_record(g_idx, results, q)
                    f.write(json.dumps({"idx": g_idx, "question": q, "scores": {r.metric_name: r.score for r in results}}) + "\n")
                    f.flush()

        run_result.finalize(self.sanitizer)
        with open(manifest_path, "w") as f:
            json.dump(run_result.to_manifest(), f, indent=2)

        if cfg.budget.track_tokens:
            self.cost_tracker.save_ledger(output_dir, cfg.experiment_id)

        self.cost_tracker.print_summary()
        if run_result.sanitizer_summary:
            logger.info(f"Sanitizer detections: {run_result.sanitizer_summary}")
        return run_result

    async def _eval_record(self, query_idx, record, metrics):
        async with self._semaphore:
            q = record.get(self.config.dataset.question_col, "")
            ctxs = record.get(self.config.dataset.context_col, [])
            a = record.get(self.config.dataset.answer_col, "")
            gt = record.get(self.config.dataset.ground_truth_col)
            clean_q, clean_a, clean_ctxs = self.sanitizer.sanitize(q, a, ctxs)  # FIX-7
            results = []
            for mc, metric in metrics:
                result = await self._score_with_retry(metric, clean_q, clean_ctxs, clean_a, gt, dict(mc.kwargs))
                results.append(result)
                self.cost_tracker.check_budget()
            return q, results

    async def _score_with_retry(self, metric, question, contexts, answer, ground_truth, kwargs):
        rc = self.config.retry
        last_error = None
        for attempt in range(rc.max_retries + 1):
            try:
                start = time.monotonic()
                result = await asyncio.wait_for(
                    metric.score_async(question=question, contexts=contexts, answer=answer, ground_truth=ground_truth, **kwargs),
                    timeout=self.config.async_cfg.timeout_seconds,
                )
                latency = time.monotonic() - start
                if hasattr(result, "metadata") and "prompt_tokens" in result.metadata:
                    self.cost_tracker.record(
                        model=str(self.config.model), metric_name=metric.name,
                        prompt_tokens=result.metadata["prompt_tokens"],
                        completion_tokens=result.metadata.get("completion_tokens", 0),
                        latency_seconds=latency, call_id=str(uuid.uuid4()),
                    )
                return result
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"{metric.name} timed out")
            except Exception as e:
                last_error = e
                if attempt < rc.max_retries:
                    wait = min(rc.backoff_base_seconds * (2 ** attempt), rc.backoff_max_seconds)
                    logger.warning(f"[{metric.name}] attempt {attempt+1} failed: {e}. Retry in {wait:.1f}s")
                    await asyncio.sleep(wait)
        return metric.error_result(str(last_error))
