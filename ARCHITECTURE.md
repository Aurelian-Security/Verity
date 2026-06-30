# ARCHITECTURE.md — Verity Evaluation Engine

**Aurelian Security | Verity v0.3.0**

> This document is the authoritative reference for Verity's system design, component responsibilities, data flow, and the engineering rationale behind its architectural decisions.

---

## 1. System Overview

Verity is a CLI-driven, asynchronous evaluation harness for RAG architectures. It decouples *generative workloads* (LLM-driven debate rounds, answer synthesis) from *scoring workloads* (statistical metrics, RAGAS grounding, safety classification) to achieve reproducible, consistent evaluation without introducing shared mutable state between runs.

The three architectural invariants that everything else flows from:

1. **Evaluation is a pipeline, not a loop.** Each evaluation item moves forward through distinct stages; no stage writes back into an upstream stage.
2. **Generative agents never touch the statistical engine directly.** All agent outputs are materialized to JSONL before metrics are computed.
3. **Distributed mode is opt-in; sync mode is the default.** A researcher with no Redis server can still run the full pipeline.

---

## 2. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OPERATOR / RESEARCHER                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  verity run / verity oversight-run
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CLI  (eval_engine/cli.py — Typer)                                      │
│  • Parses flags and YAML config                                         │
│  • Validates EvalConfig via Pydantic v2                                 │
│  • Routes to EvalRunner (metrics) or OversightRunner (debate pipeline)  │
│  • Handles verity compare, verity manifest, verity validate-config      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  EvalConfig (typed, validated)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CONFIG LAYER  (eval_engine/config.py — Pydantic v2)                   │
│  • EvalConfig: dataset path, metrics list, model, architecture,         │
│    budget ceiling, concurrency, seed, track flag, agent params          │
│  • metrics field min_length=0: oversight runs don't require metrics     │
│  • Validates all fields at parse time — invalid configs crash early     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴──────────────┐
                    │                           │
                    ▼                           ▼
      ┌─────────────────────┐    ┌───────────────────────────┐
      │  EVAL RUNNER        │    │  OVERSIGHT RUNNER          │
      │  (runner.py)        │    │  (orchestration/           │
      │  Metric evaluation  │    │   oversight_runner.py)     │
      │  asyncio semaphore  │    │  Dataset-level debate exec │
      │  retry / JSONL      │    │  asyncio semaphore         │
      │  seed control       │    │  CostTracker + StatEngine  │
      │  reproducibility    │    │  wired in                  │
      └──────────┬──────────┘    └──────────────┬────────────┘
                 │                              │
                 └──────────┬───────────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
         ┌──────────────┐  ┌────────────────────┐
         │  SYNC MODE   │  │  CELERY / REDIS     │
         │  (default)   │  │  DISTRIBUTED MODE   │
         │  In-process  │  │  (--celery flag)    │
         └──────┬───────┘  └────────┬────────────┘
                └──────────┬────────┘
                           │  Evaluation items
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MULTI-AGENT OVERSIGHT PIPELINE  (eval_engine/agents/)                  │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                         │
│   │ PROPOSER │───▶│  CRITIC  │───▶│  JUDGE   │                         │
│   │ Agent A  │    │ Agent B  │    │ Agent C  │                         │
│   │(generate │    │(challenge│    │(verdict: │                          │
│   │ answer)  │    │ grounding│    │ Pass /   │                          │
│   │          │    │ + RH     │    │ Cond /   │                          │
│   │          │    │ detect.) │    │ Fail)    │                          │
│   └──────────┘    └──────────┘    └──────────┘                         │
│                                                                         │
│  All content sanitized by sanitizer.py before LLM submission            │
│  CostTracker wraps every API call — aborts at budget ceiling            │
│  Per-debate trace JSON written to debates/ alongside results            │
│  All agent outputs materialized to JSONL before scoring begins          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  Materialized agent outputs (JSONL)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  METRIC ENGINE  (eval_engine/metrics/)                                  │
│  • 74 registered metrics (21 core + 4 rankers + 47 expansion + 2 Tier 3 scaffolds)  │
│  • BaseMetric ABC enforces .score() interface via Pydantic MetricResult │
│  • Metrics read from JSONL — never call agents or mutate state          │
│  • registry.register() supports runtime plugin injection                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  Per-item MetricResult objects
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STATISTICAL ENGINE  (eval_engine/statistics.py — StatEngine)           │
│  • Wilcoxon, Mann-Whitney, paired t-test (significance)                 │
│  • Cohen's d, Cliff's delta (effect size)                               │
│  • Pearson, Spearman (correlation)                                      │
│  • Shapiro-Wilk normality, descriptive stats                            │
│  • pre_post_bundle(): 7-test suite for one results table row            │
│  • Receives debate batch scores from OversightRunner                    │
│  • StatReport: serializes to JSON, prints rich summary table            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  REPRODUCIBILITY LAYER  (eval_engine/reproducibility.py,                │
│                          eval_engine/dataset_manifest.py,               │
│                          eval_engine/comparison.py)                     │
│  • ReproducibilityBundle: seed, git hash, Python version, all deps      │
│  • DatasetManifest: SHA-256 content hash, schema, lineage chain         │
│  • compare_run_manifests(): score deltas, dataset identity diff         │
│  • Written when --track flag set; verity compare/manifest CLI commands  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT LAYER  (outputs/{experiment_id}/)                               │
│  • results.jsonl              — per-item scores, token counts           │
│  • oversight_results.jsonl    — per-item agent verdicts                 │
│  • oversight_manifest.json    — run config snapshot, seeds, timestamp   │
│  • oversight_stats.json       — StatEngine output for debate batch      │
│  • debates/{query_id}.json    — full A→B→C trace per debate item        │
│  • manifest.json              — run config snapshot (metric runs)       │
│  • cost_ledger_{id}.jsonl     — per-call token + cost accounting        │
│  • reproducibility.json       — seed, git hash, deps (when --track)     │
│  • dataset_manifest.json      — SHA-256 + lineage (when --track)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Verity/
├── eval_engine/
│   ├── cli.py                  ← Typer CLI (6 commands)
│   ├── config.py               ← Pydantic v2 EvalConfig + YAML loader
│   ├── runner.py               ← Async EvalRunner: semaphore, retry, JSONL, seed control
│   ├── cost_tracker.py         ← Per-call token accounting; enforces budget ceiling
│   ├── sanitizer.py            ← Pattern-based prompt injection sanitization
│   ├── statistics.py           ← StatEngine: 9 methods + StatReport
│   ├── schemas.py              ← RetrievalCase, RetrievalResult, GraphSnapshot
│   ├── reproducibility.py      ← ReproducibilityBundle: seed, git, deps capture
│   ├── dataset_manifest.py     ← DatasetManifest: SHA-256 hash + lineage tracking
│   ├── comparison.py           ← compare_run_manifests(): run diff engine
│   │
│   ├── metrics/
│   │   ├── __init__.py         ← MetricsRegistry: 74 registered metrics
│   │   ├── base.py             ← BaseMetric ABC + MetricResult dataclass
│   │   ├── retrieval_metrics.py     ← Recall@K, MRR, NDCG@K
│   │   ├── retrieval_rankers.py     ← BM25, RRF, MMR, LTR heuristic (v0.5.1)
│   │   ├── graph_metrics.py         ← Compression delta, dedup delta, entity coverage
│   │   ├── ragas_adapter.py         ← RAGAS dataset conversion
│   │   ├── ragas_runner.py          ← RAGAS execution + LLM backend config
│   │   ├── ragas_grounding.py       ← Single-phase RAGAS grounding
│   │   ├── ragas_consolidation_delta.py  ← Pre/post consolidation delta
│   │   ├── llamaguard.py            ← LlamaGuard-3-8B safety classifier
│   │   ├── calibration.py           ← ECE + Brier Score
│   │   ├── hallucination.py         ← Claim-level hallucination rate
│   │   ├── trust_score.py           ← Composite trust score
│   │   ├── multi_session_persistence.py  ← N-cycle forgetting curve
│   │   ├── per_stage_ablation.py    ← Stage contribution measurement
│   │   ├── threshold_compute_budget.py   ← Compute budget curve
│   │   ├── single_session_poisoning.py   ← Adversarial poisoning test
│   │   ├── query_perturbation.py    ← Query perturbation robustness
│   │   ├── consistency.py           ← [Scaffold Tier 2]
│   │   ├── constitutional_eval.py   ← [Scaffold Tier 2]
│   │   ├── model_written_eval.py    ← [Scaffold Tier 2]
│   │   ├── source_reliability.py    ← [Scaffold Tier 2]
│   │   ├── goal_misgeneralization.py ← [Scaffold Tier 3]
│   │   ├── deceptive_alignment.py   ← [Scaffold Tier 3]
│   │   └── algorithm_expansion.py   ← 47-algorithm expansion registry (v0.5.0)
│   │
│   ├── agents/
│   │   ├── agent_base.py       ← Shared base: real/dry-run dispatch, trace logging
│   │   ├── proposer.py         ← Agent A: grounded answer generation
│   │   ├── critic.py           ← Agent B: reward hacking detection
│   │   └── judge.py            ← Agent C: verdict synthesis
│   │
│   ├── orchestration/
│   │   ├── debate_round.py     ← Single A→B→C pipeline unit + DebateResult
│   │   ├── oversight_runner.py ← Dataset-level debate execution + stats + manifest
│   │   └── celery_tasks.py     ← Celery task definitions + sync fallback
│   │
│   └── tests/
│       ├── test_eval_engine.py         ← Core metrics, sanitizer, budget tracker (31)
│       ├── test_statistics.py          ← StatEngine edge cases (22)
│       ├── test_phase2_metrics.py      ← Ablation, poisoning, perturbation (19)
│       ├── test_phase3_pipeline.py     ← Agent pipeline, debate round (32)
│       ├── test_oversight_runner.py    ← OversightRunner, OversightRunResult, CLI (21)
│       ├── test_tier1_metrics.py       ← Calibration, hallucination, trust, persistence (37)
│       ├── test_scaffolds.py           ← Scaffold registry and interface contracts (50)
│       └── test_phase4.py              ← Reproducibility, dataset manifest, comparison (33)
│
├── configs/
│   ├── consolidation_eval_example.yaml ← Example consolidation evaluation config
│   └── centralized_baseline.yaml       ← Centralized RAG baseline config
│
├── datasets/                   ← User-provided datasets (gitignored)
├── outputs/                    ← Generated results (gitignored)
├── docker-compose.yml          ← Redis + Celery worker + Flower monitor
├── Dockerfile.worker           ← Worker container image
├── pyproject.toml              ← Build system, deps, dev tools, pytest config
├── .env.example                ← Environment variable template
├── README.md
├── ARCHITECTURE.md             ← This document
├── STATUS.md                   ← Implementation maturity matrix
├── SECURITY.md                 ← Threat model + responsible disclosure
├── CHANGELOG.md                ← Version history
└── LICENSE                     ← MIT
```

---

## 4. Component Deep-Dives

### 4.1 CLI → Config → Runner

The CLI (`cli.py`) is a pure dispatch layer. It parses input, instantiates `EvalConfig`, and routes to either `EvalRunner` (metric evaluation via `verity run`) or `OversightRunner` (debate pipeline via `verity oversight-run`). No business logic lives in `cli.py`.

`EvalConfig` (Pydantic v2) validates all parameters at instantiation time. The `metrics` field accepts an empty list — oversight runs do not require metrics. The `seed` field (default: 42) flows through to `set_global_seed()` at run start. The `track` boolean gates reproducibility bundle and dataset manifest generation.

`EvalRunner` uses `asyncio.Semaphore(config.concurrency)` to bound parallel API calls. Each item is retried up to 3× with exponential backoff on `RateLimitError`. Items are streamed to `results.jsonl` as they complete.

### 4.2 OversightRunner

`OversightRunner` is the dataset-level wrapper around `DebateRound`. It closes the wiring gap between the debate pipeline and the rest of the platform:

- Accepts a dataset and `EvalConfig`
- Dispatches each item through `DebateRound` with async semaphore concurrency
- Feeds per-call token counts into the main `CostTracker` — `BudgetExceededError` applies to oversight runs
- Collects `DebateResult` objects and passes safety/accuracy score arrays to `StatEngine`
- Writes `oversight_results.jsonl`, `oversight_manifest.json`, `oversight_stats.json`, and per-debate trace JSONs

### 4.3 Multi-Agent Oversight Pipeline

The Proposer → Critic → Judge pipeline is the principal oversight mechanism and primary defense against reward hacking.

- **Proposer**: Given a query and retrieved contexts, generates a candidate answer. No network access, no tool calls. Output is a structured typed object.
- **Critic**: Receives the query, contexts, and Proposer output. Runs a zero-cost sycophancy pre-screen before the LLM deep audit. Detects: sycophantic openers, confidence inflation, factual drift, metric gaming.
- **Judge**: Receives all prior context plus the Critic's critique. Issues a verdict (`Pass`, `Conditional`, `Fail`) and scores. The verdict gates whether the item's scores are included in the statistical report.

This structure enforces the OWASP LLM08 (Excessive Agency) mitigation: the Proposer has no ability to trigger downstream actions. Its output must survive Critic review and Judge approval before it influences any scored result.

**Dry-run mode** (`DebateRound(dry_run=True)`) stubs all three agent calls with deterministic fixture responses, enabling zero-cost CI testing and local demos without API keys.

### 4.4 Sanitizer

`sanitizer.py` applies prompt injection defenses to any text passed to the Judge agent. The sanitizer strips common injection patterns (role-switching, ignore-instructions, persona override, score manipulation) before Judge submission. This is not a complete defense — see `SECURITY.md` for the full threat model and residual risks.

### 4.5 Statistical Engine

`StatEngine` computes all statistics after metric scoring is complete, operating only on arrays of floats. In v0.3.0, it also receives debate batch scores from `OversightRunner` for safety/accuracy analysis.

Edge case handling is explicit:
- **Cohen's d with N < 10**: logs warning, returns `None`
- **Uniform distribution (std=0)**: logs warning, returns `None`
- **Wilcoxon with tied ranks**: uses tie-correction with `RuntimeWarning`
- **Pearson on constant input**: returns `ConstantInputWarning`, handled gracefully

### 4.6 Reproducibility Layer

Three modules added in v0.3.0:

- **`reproducibility.py`**: Captures seed, Python version, platform, git hash/branch/dirty flag, all installed package versions, and config snapshot into `reproducibility.json`. `set_global_seed()` seeds Python random, numpy, and torch.
- **`dataset_manifest.py`**: Content-addresses datasets via SHA-256. Supports full transformation lineage for poisoning/consolidation pipelines. `version_tag` field supports `v1.0`-style benchmark versioning.
- **`comparison.py`**: Diffs two run manifests — metric score deltas, verdict distribution changes, dataset SHA-256 identity verification, git hash comparison.

### 4.7 Cost Tracker

`CostTracker` maintains a running token ledger across all API calls, wired into both `EvalRunner` and the debate pipeline. Before each call, it checks whether spend would exceed `config.budget`. If yes, the call is aborted with `BudgetExceededError`. A JSONL ledger is written per run.

### 4.8 Distributed Mode (Redis + Celery)

When `docker compose up -d` is running, both `EvalRunner` and `OversightRunner` detect the Redis URL and route through `celery_tasks.py`. The sync fallback provides identical semantics without Redis. `docker compose --profile monitoring up -d` enables the Flower task monitor at `:5555`.

---

## 5. Decoupling Rationale

The separation between generative workloads and the scoring engine is the core architectural decision.

| Approach | Problem |
|---|---|
| Agents write metric scores directly | Scores become entangled with generation context; reward hacking risk |
| Streaming scores from agent outputs | Non-deterministic ordering; statistical tests require complete arrays |
| Shared in-memory score store | Fails under distributed worker scaling; concurrency bugs |
| Single-pass eval (generate + score in one call) | Cannot replay scoring with different metrics without re-spending API budget |

Materializing agent outputs to JSONL before scoring means:
- Metrics can be re-computed from the same outputs without re-calling the LLM
- The statistical engine is fully unit-testable without any LLM dependency
- A corrupted or biased agent run can be detected and excluded before it influences published statistics

---

## 6. Provider Abstraction

| Provider | Install Extra | Env Var |
|---|---|---|
| Anthropic (default) | `pip install -e .` | `ANTHROPIC_API_KEY` |
| OpenAI | `pip install -e ".[openai-backend]"` | `OPENAI_API_KEY` |
| Ollama (local, air-gapped) | `pip install -e ".[ollama-backend]"` | None |

The RAGAS evaluation backend must be configured separately via `configure_ragas_llm()` in `ragas_runner.py` — RAGAS defaults to OpenAI regardless of `config.model` if not explicitly set.

---

## 7. Known Architectural Limitations

- **No GUI.** Verity is CLI + SDK only.
- **LlamaGuard requires local GPU.** The `llamaguard_safety` metric uses `transformers` + `torch` and requires 16GB+ VRAM. Excluded from standard CI.
- **Distributed mode not load-tested.** Celery + Redis is implemented and functional; large-scale concurrency has not been benchmarked.
- **Judge sanitizer is partial.** Prompt injection resistance is a known open problem. See `SECURITY.md`.
- **mypy union-attr on Anthropic SDK.** The Anthropic content block union type causes mypy to flag `.text` access. Suppressed with `# type: ignore[union-attr]`; runtime behavior is correct.
- **`multi_session_persistence` has limited test coverage.** Longitudinal evaluation across multiple sessions requires a persistent dataset fixture not yet fully implemented in the test suite.

---

*Verity is reproducible open-source research infrastructure for RAG evaluation and adversarial robustness testing. It is not a finished enterprise product.*

*© 2026 Aurelian Security — MIT License*
