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
- Example dataset in `datasets/` for zero-setup quick start
- GitHub Actions CI workflow

---

## [0.3.0] — 2026-06

### Added
- **`reproducibility.py`** — `ReproducibilityBundle`: captures seed, Python version, platform, git hash/branch/dirty flag, all installed package versions, and config snapshot per run. `set_global_seed()` seeds Python random + numpy + torch.
- **`dataset_manifest.py`** — `DatasetManifest`: SHA-256 content-addresses datasets, records row count, columns, schema, version tag, and full transformation lineage chain for poisoning/consolidation pipelines.
- **`comparison.py`** — `compare_run_manifests()`: diffs two run manifests — metric score deltas, verdict distribution changes, dataset SHA-256 identity verification, git hash comparison. Saves `compare_A_vs_B.json`.
- **`verity compare` CLI command** — `--run-a` and `--run-b` flags; accepts experiment IDs or direct paths.
- **`verity manifest` CLI command** — standalone dataset inspection: SHA-256 hash, schema, version tag, optional `--output` to save manifest.
- **`--seed` flag** on `verity run` — random seed flows through `EvalConfig` into all random sampling, poisoning index selection, and perturbation operations.
- **`--track` flag** on `verity run` — writes `reproducibility.json` + `dataset_manifest.json` alongside normal outputs.
- **`seed` and `track` fields** on `EvalConfig` — configurable in YAML config or via CLI flags.
- **33 new tests** (`tests/test_phase4.py`) covering reproducibility bundle, dataset manifest, comparison, and CLI flags.

### Changed
- `runner.py` — `set_global_seed()` called at run start; seed flows into `random.sample()` for dataset subsampling; reproducibility bundle and dataset manifest written when `config.track` is True.
- `config.py` — `seed: int = 42` and `track: bool = False` fields added to `EvalConfig`.
- `pyproject.toml` — version bumped to `0.3.0`.

---

## [0.2.1] — 2026-06

### Added
- **`OversightRunner`** (`orchestration/oversight_runner.py`) — dataset-level execution wrapper around `DebateRound`. Async with semaphore concurrency. Feeds into `CostTracker` and `StatEngine`. Writes `oversight_results.jsonl`, `oversight_manifest.json`, `oversight_stats.json`, and per-debate trace JSONs.
- **`verity oversight-run` CLI command** — full flag set: `--dry-run`, `--proposer-model`, `--critic-model`, `--judge-model`, `--recall-at-k`, `--budget`, `--concurrency`, `--celery`.
- **Cost tracking wired into debate pipeline** — per-agent token counts now feed the main `CostTracker`; `BudgetExceededError` applies to oversight runs.
- **StatEngine integration for debate batches** — `OversightRunner` passes safety and accuracy score arrays to `StatEngine`.
- **Per-debate trace JSON output** — full A→B→C trace written per item to `debates/` subdirectory.
- **`OversightRunResult`** — typed result object with per-item verdicts, aggregate stats, reward hacking rate, verdict distribution.
- **21 new tests** (`tests/test_oversight_runner.py`) covering `OversightRunResult`, `OversightRunner`, CLI, and exports.
- **Documentation suite** — `ARCHITECTURE.md`, `SECURITY.md`, `STATUS.md`, `CHANGELOG.md`.

### Changed
- `orchestration/__init__.py` — exports `OversightRunner`, `OversightRunResult`, `DebateRound`, `DebateResult`.
- `eval_engine/__init__.py` — all four orchestration types importable from top-level `eval_engine`.
- `cli.py` — `verity oversight-run` added; `list-metrics` scope map updated.
- `config.py` — `metrics` field `min_length` relaxed to 0 (oversight runs do not require metrics).

---

## [0.2.0] — 2026-06

### Added
- **Multi-agent oversight pipeline**: Proposer → Critic → Judge with `Pass / Conditional / Fail` verdict and `reward_hacking_confirmed` flag.
- **Sycophancy pre-screen** in Critic agent (zero cost, rule-based).
- **Dry-run mode** (`DebateRound(dry_run=True)`): deterministic fixture responses, zero API cost, CI-safe.
- **Cost tracker**: per-call token accounting with `BudgetExceededError` on ceiling breach.
- **Prompt injection sanitizer** (`sanitizer.py`): pattern-based sanitization applied to all Judge inputs.
- **Celery + Redis distributed mode**: `docker-compose.yml` with Redis broker, worker containers, and optional Flower monitor.
- **Sync fallback**: full pipeline without Redis dependency (default for local use and dry-run).
- **StatEngine**: Wilcoxon, Mann-Whitney, paired t-test, Cohen's d, Cliff's delta, Pearson, Spearman, Shapiro-Wilk, descriptive, `pre_post_bundle()`.
- **StatReport**: JSON serialization + Rich summary table.
- **Run manifest** (`manifest.json`): config snapshot, timestamp, cost summary per run.
- **6 scaffolded metric interfaces**: `consistency`, `constitutional_eval`, `model_written_eval`, `source_reliability`, `goal_misgeneralization`, `deceptive_alignment`.
- **Tier 1 trust metrics**: `calibration` (ECE + Brier), `hallucination_rate` (claim-level), `trust_score` (weighted composite), `multi_session_persistence` (N-cycle forgetting curve).
- **`verity validate-config`** and **`verity list-metrics`** CLI subcommands.
- Provider abstraction: `anthropic-backend`, `openai-backend`, `ollama-backend` optional extras.
- Renamed project from `eval-engine` to **Verity**; CLI entrypoint changed from `eval-engine` to `verity`.

### Fixed
- Cohen's d returning `NaN` on uniform-distribution samples — now returns `None` with logged warning.
- Wilcoxon rank-sum crash on tied ranks — now uses tie-correction with `RuntimeWarning`.
- Redis `maxmemory` not set — now configurable via `REDIS_MAX_MEMORY` env var (default `512mb`).

---

## [0.1.0] — 2026-05

### Added
- Initial repository: `eval_engine/` package with `cli.py`, `config.py`, `runner.py`, `schemas.py`.
- 17 implemented metrics: `ndcg`, `recall_at_k`, `mean_reciprocal_rank`, `ragas_grounding`, `ragas_consolidation_delta`, `compression_delta`, `deduplication_delta`, `entity_coverage`, `llamaguard_safety`, `per_stage_ablation`, `threshold_compute_budget`, `single_session_poisoning`, `query_perturbation`, `calibration`, `hallucination_rate`, `trust_score`, `multi_session_persistence`.
- `BaseMetric` ABC and metric plugin registry.
- Async `EvalRunner` with `asyncio.Semaphore` concurrency control.
- JSONL streaming output to `outputs/{experiment_id}/`.
- `pyproject.toml` build system with `verity` CLI entrypoint.
- `Dockerfile.worker` and `docker-compose.yml` (Redis + Celery + Flower).
- `.env.example`, `.gitignore`, `.dockerignore`.
- MIT License.

---

*© 2026 Aurelian Security — MIT License*
