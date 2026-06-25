# CHANGELOG.md — Verity

All notable changes to Verity are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format. Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- `--redact-contexts` flag to strip retrieved context text from JSONL outputs
- Bootstrap confidence intervals in StatEngine
- Multiple comparison correction (Bonferroni / Benjamini-Hochberg) in StatReport
- Coverage reporting in CI
- mypy strict mode (blocked by Anthropic SDK union-attr types)

---

## [0.2.1] — 2026-06

### Added
- **`OversightRunner`** (`orchestration/oversight_runner.py`) — dataset-level execution wrapper around `DebateRound`. Async with semaphore concurrency. Feeds into `CostTracker` and `StatEngine`. Writes `oversight_results.jsonl`, `oversight_manifest.json`, `oversight_stats.json`, and per-debate trace JSONs.
- **`verity oversight-run` CLI command** — full flag set: `--dry-run`, `--proposer-model`, `--critic-model`, `--judge-model`, `--recall-at-k`, `--budget`, `--concurrency`, `--celery`
- **Cost tracking wired into debate pipeline** — per-agent token counts now feed the main `CostTracker`; `BudgetExceededError` applies to oversight runs
- **StatEngine integration for debate batches** — `OversightRunner` passes safety and accuracy score arrays to `StatEngine` for Wilcoxon/Cohen's d analysis across a run
- **Per-debate trace JSON output** — full A→B→C trace written per item alongside `oversight_results.jsonl`
- **`OversightRunResult`** schema — typed result object with per-item verdicts, aggregate stats, cost summary
- **21 new tests** (`tests/test_oversight_runner.py`) covering `OversightRunResult`, `OversightRunner`, CLI, and exports
- **Full documentation suite** — `ARCHITECTURE.md`, `SECURITY.md`, `STATUS.md`, `REPRODUCIBILITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — pytest, ruff, mypy (non-blocking), build check

### Changed
- `orchestration/__init__.py` — updated to export `OversightRunner`, `OversightRunResult`, `DebateRound`, `DebateResult`
- `eval_engine/__init__.py` — all four orchestration types now importable from top-level `eval_engine`
- `cli.py` — `verity oversight-run` added; `list-metrics` scope map cleaned up
- `config.py` — `metrics` field `min_length` relaxed to 0 (oversight runs do not require metrics)
- CI mypy job set to non-blocking (`|| true`) pending resolution of Anthropic SDK union-attr typing

### Fixed
- Anthropic SDK `union-attr` mypy errors suppressed with targeted `# type: ignore[union-attr]` at call sites in `agent_base.py`, `hallucination.py`, `query_perturbation.py`
- Override signature mypy errors suppressed in `proposer.py`, `critic.py`, `judge.py`
- Celery `.s()` attr-defined mypy error suppressed in `celery_tasks.py`
- ruff E401 (multiple imports on one line) fixed across `runner.py`, `metrics/__init__.py`, `agents/proposer.py`
- ruff F541 (f-string without placeholders) fixed in `calibration.py`, `hallucination.py`
- ruff F841 (unused variable `instance`) fixed in `metrics/__init__.py`

---

## [0.2.0] — 2026-06

### Added
- **Multi-agent oversight pipeline**: Proposer → Critic → Judge with `Pass / Conditional / Fail` verdict
- **Sycophancy pre-screen** in Critic agent
- **Dry-run mode** (`DebateRound(dry_run=True)`): deterministic fixture responses, zero API cost, CI-safe
- **Cost tracker**: per-call token accounting with `BudgetExceededError` on ceiling breach
- **Prompt injection sanitizer** (`sanitizer.py`): pattern-based sanitization applied to all Judge inputs
- **Celery + Redis distributed mode**: `docker-compose.yml` with Redis broker, worker containers, and optional Flower monitor
- **Sync fallback** (`sync_fallback.py`): full pipeline without Redis dependency
- **StatEngine**: Wilcoxon rank-sum, Cohen's d (with small-N and zero-variance guards), Pearson correlation, ECE
- **StatReport**: JSON serialization + Rich summary table
- **Run manifest** (`manifest.json`): config snapshot, dataset SHA-256, timestamp, cost summary per run
- **6 scaffolded metric interfaces**: `consistency`, `constitutional_eval`, `model_written_eval`, `source_reliability`, `goal_misgeneralization`, `deceptive_alignment`
- **`verity validate-config`** CLI subcommand
- **`verity list-metrics`** CLI subcommand with scaffold/implemented distinction
- Provider abstraction: `anthropic-backend`, `openai-backend`, `ollama-backend` optional extras
- `examples/` directory with `mock_debate/`, `ragas_grounding/`, `poisoning_test/`, `ablation_test/`, `budget_sweep/`

### Changed
- Renamed project from `eval-engine` to **Verity** and moved to `Aurelian-Security/Verity` repository
- CLI entrypoint changed from `eval-engine` to `verity`
- `EvalConfig` migrated from Pydantic v1 to Pydantic v2 (breaking change)

### Fixed
- Cohen's d returning `NaN` on uniform-distribution samples — now returns `None` with logged warning
- Wilcoxon rank-sum crash on tied ranks — now uses `method='approx'` with tie-correction warning
- Redis `maxmemory` not set — now configurable via `REDIS_MAX_MEMORY` env var (default `512mb`)

---

## [0.1.0] — 2026-05

### Added
- Initial repository: `eval_engine/` package with `cli.py`, `config.py`, `runner.py`, `schemas.py`
- 17 implemented metrics: `ndcg`, `recall_at_k`, `mean_reciprocal_rank`, `ragas_grounding`, `ragas_consolidation_delta`, `compression_delta`, `deduplication_delta`, `entity_coverage`, `llamaguard_safety`, `per_stage_ablation`, `threshold_compute_budget`, `single_session_poisoning`, `query_perturbation`, `calibration`, `hallucination_rate`, `trust_score`, `multi_session_persistence`
- `BaseMetric` ABC and metric plugin registry
- Async `EvalRunner` with `asyncio.Semaphore` concurrency control
- JSONL streaming output to `outputs/{experiment_id}/`
- `pyproject.toml` build system with `verity` CLI entrypoint
- `Dockerfile.worker` and `docker-compose.yml` (Redis + Celery + Flower)
- `.env.example`, `.gitignore`, `.dockerignore`
- MIT License

---

*© 2026 Aurelian Security — MIT License*
