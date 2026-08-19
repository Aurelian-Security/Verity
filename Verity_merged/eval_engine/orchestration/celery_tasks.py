"""
eval_engine/orchestration/celery_tasks.py

Celery task definitions for distributed debate pipeline.

Each debate round is dispatched as a Celery task, enabling:
  - Parallel processing across multiple workers
  - Queue-based backpressure (don't overwhelm the API)
  - Automatic retry on transient failures
  - Task result storage in Redis backend

Architecture:
    CLI / EvalRunner
        → dispatch_debate_batch() [submits N tasks to Redis queue]
            → run_debate_task() [Celery worker executes one DebateRound]
                → DebateRound.run() [A→B→C pipeline]
                    → result stored in Redis
        → collect_debate_results() [polls until all tasks complete]

Setup (run these before using):
    # Terminal 1: Start Redis
    docker run -d -p 6379:6379 redis:7-alpine

    # Terminal 2: Start Celery worker
    celery -A eval_engine.orchestration.celery_tasks worker \
           --loglevel=info --concurrency=4

    # Terminal 3: Run pipeline
    verity oversight-run --config configs/consolidation_rag_paper1.yaml

RESOURCE NOTE:
    Worker concurrency=4 means 4 debate rounds in parallel.
    Each round makes 3 API calls (proposer, critic, judge).
    Total concurrent API calls = 4 × 3 = 12.
    Set concurrency based on your Anthropic rate limits.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app setup
# ---------------------------------------------------------------------------

def _make_celery_app():
    """
    Create Celery app. Lazy import so Celery is optional for users
    who only use the sync pipeline (DebateRound directly).
    """
    try:
        from celery import Celery
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        app = Celery(
            "eval_engine",
            broker=redis_url,
            backend=redis_url,
        )
        app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            task_track_started=True,
            task_acks_late=True,            # Re-queue on worker crash
            worker_prefetch_multiplier=1,   # One task per worker at a time
            result_expires=3600,            # Results expire after 1 hour
        )
        return app
    except ImportError:
        return None


celery_app = _make_celery_app()


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

def run_debate_task(
    query_id: str,
    query: str,
    contexts: list[str],
    dry_run: bool = False,
    proposer_model: str | None = None,
    critic_model: str | None = None,
    judge_model: str | None = None,
    recall_at_k: int = 5,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Execute one debate round. Called by Celery worker.

    Returns serializable dict (DebateResult.to_dict()).
    Saves individual result JSON if output_dir provided.
    """
    from eval_engine.orchestration.debate_round import DebateRound

    logger.info(f"[CeleryTask] Starting debate for query_id={query_id}")

    round_runner = DebateRound(
        dry_run=dry_run,
        proposer_model=proposer_model,
        critic_model=critic_model,
        judge_model=judge_model,
    )

    result = round_runner.run(
        query_id=query_id,
        query=query,
        contexts=contexts,
        recall_at_k=recall_at_k,
    )

    result_dict = result.to_dict()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"debate_{query_id}.json"
        with open(path, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"[CeleryTask] Result saved: {path}")

    return result_dict


# Register as Celery task if Celery is available
if celery_app is not None:
    run_debate_task = celery_app.task(
        bind=True,
        name="eval_engine.debate",
        max_retries=3,
        default_retry_delay=5,
        autoretry_for=(Exception,),
        retry_backoff=True,
    )(run_debate_task)


# ---------------------------------------------------------------------------
# Batch dispatch and collection
# ---------------------------------------------------------------------------

def dispatch_debate_batch(
    records: list[dict[str, Any]],
    dry_run: bool = False,
    proposer_model: str | None = None,
    critic_model: str | None = None,
    judge_model: str | None = None,
    recall_at_k: int = 5,
    output_dir: str | None = None,
    use_celery: bool = True,
) -> list[Any]:
    """
    Dispatch a batch of debate tasks.

    Args:
        records:   List of dicts with keys: query_id, query, contexts
        use_celery: If False, runs synchronously (no Redis required).
                    Use for local testing without Docker.

    Returns:
        List of AsyncResult handles (Celery) or DebateResult dicts (sync).
    """
    if not use_celery or celery_app is None:
        logger.info(f"[Dispatch] Sync mode: {len(records)} debates")
        return _run_sync(
            records, dry_run, proposer_model, critic_model,
            judge_model, recall_at_k, output_dir
        )

    try:
        from celery import group
        logger.info(f"[Dispatch] Celery mode: {len(records)} debates → Redis queue")
        tasks = group(
            run_debate_task.s(
                query_id=r["query_id"],
                query=r["query"],
                contexts=r.get("contexts", []),
                dry_run=dry_run,
                proposer_model=proposer_model,
                critic_model=critic_model,
                judge_model=judge_model,
                recall_at_k=recall_at_k,
                output_dir=output_dir,
            )
            for r in records
        )
        result = tasks.apply_async()
        return result.results

    except Exception as e:
        logger.warning(
            f"[Dispatch] Celery dispatch failed: {e}. "
            f"Falling back to sync mode."
        )
        return _run_sync(
            records, dry_run, proposer_model, critic_model,
            judge_model, recall_at_k, output_dir
        )


def collect_debate_results(
    task_handles: list[Any],
    timeout_seconds: float = 300.0,
    poll_interval: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Poll until all Celery tasks complete or timeout.

    Args:
        task_handles:     AsyncResult list from dispatch_debate_batch()
        timeout_seconds:  Max wait time
        poll_interval:    Seconds between polls

    Returns:
        List of DebateResult dicts (in submission order).
    """
    # If sync mode (plain dicts), return immediately
    if task_handles and isinstance(task_handles[0], dict):
        return task_handles

    start = time.monotonic()
    results = [None] * len(task_handles)
    pending = set(range(len(task_handles)))

    while pending and (time.monotonic() - start) < timeout_seconds:
        newly_done = set()
        for i in list(pending):
            handle = task_handles[i]
            try:
                if handle.ready():
                    results[i] = handle.get(timeout=10)
                    newly_done.add(i)
            except Exception as e:
                logger.error(f"[Collect] Task {i} failed: {e}")
                results[i] = {"error": str(e), "query_id": f"task_{i}"}
                newly_done.add(i)

        pending -= newly_done
        if pending:
            logger.info(f"[Collect] {len(pending)} tasks remaining...")
            time.sleep(poll_interval)

    if pending:
        logger.warning(
            f"[Collect] Timeout: {len(pending)} tasks did not complete "
            f"within {timeout_seconds}s"
        )
        for i in pending:
            results[i] = {"error": "timeout", "query_id": f"task_{i}"}

    return [r for r in results if r is not None]


def _run_sync(
    records: list[dict[str, Any]],
    dry_run: bool,
    proposer_model: str | None,
    critic_model: str | None,
    judge_model: str | None,
    recall_at_k: int,
    output_dir: str | None,
) -> list[dict[str, Any]]:
    """Synchronous fallback — runs debates sequentially."""
    results = []
    for r in records:
        result = run_debate_task(
            query_id=r["query_id"],
            query=r["query"],
            contexts=r.get("contexts", []),
            dry_run=dry_run,
            proposer_model=proposer_model,
            critic_model=critic_model,
            judge_model=judge_model,
            recall_at_k=recall_at_k,
            output_dir=output_dir,
        )
        results.append(result if isinstance(result, dict) else result.to_dict() if hasattr(result, 'to_dict') else result)
    return results
