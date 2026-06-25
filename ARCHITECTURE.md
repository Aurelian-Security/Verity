# ARCHITECTURE.md — Verity Evaluation Engine

**Aurelian Security | Verity v0.2.1**

> This document is the authoritative reference for Verity's system design, component responsibilities, data flow, and the engineering rationale behind its architectural decisions.

---

## 1. System Overview

Verity is a CLI-driven, asynchronous evaluation harness for RAG architectures. It decouples *generative workloads* (LLM-driven debate rounds, answer synthesis) from *scoring workloads* (statistical metrics, RAGAS grounding, safety classification) to achieve reproducible, enterprise-scale evaluation without introducing shared mutable state between runs.

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
│  • Both runners support --celery flag for distributed dispatch          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  EvalConfig (typed, validated)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CONFIG LAYER  (eval_engine/config.py — Pydantic v2)                   │
│  • EvalConfig: dataset path, metrics list, model, architecture,         │
│    budget ceiling, concurrency, poison_ratio thresholds, agent params   │
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
                           │  Evaluation items (typed EvalRecord)
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MULTI-AGENT OVERSIGHT PIPELINE  (eval_engine/agents/)                  │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                         │
│   │ PROPOSER │───▶│  CRITIC  │───▶│  JUDGE   │                         │
│   │ Agent    │    │ Agent    │    │ Agent    │                          │
│   │(generate │    │(challenge│    │(verdict: │                          │
│   │ answer)  │    │ grounding│    │ Pass /   │                          │
│   │          │    │ + safety │    │ Cond /   │                          │
│   │          │    │ + syco-  │    │ Fail)    │                          │
│   │          │    │ phancy   │    │          │                          │
│   └──────────┘    └──────────┘    └──────────┘                         │
│                                                                         │
│  All agent calls sanitized by sanitizer.py before LLM submission        │
│  CostTracker wraps every API call — aborts at budget ceiling            │
│  Per-debate trace JSON written alongside results                        │
│  All agent outputs written to JSONL before scoring begins               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  Materialized agent outputs (JSONL)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  METRIC ENGINE  (eval_engine/metrics/)                                  │
│  • 23 registered metrics (17 implemented, 6 scaffolded)                 │
│  • BaseMetric ABC enforces .score() interface via Pydantic MetricResult │
│  • Metrics read from JSONL — never call agents or mutate state          │
│  • registry.register() supports runtime plugin injection                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │  Per-item MetricResult objects
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STATISTICAL ENGINE  (eval_engine/statistics.py — StatEngine)           │
│  • Wilcoxon rank-sum (non-parametric, pre/post consolidation delta)     │
│  • Cohen's d (effect size; guarded against small-N and uniform dist.)  │
│  • Pearson correlation (score vs. context length)                       │
│  • ECE (calibration error)                                              │
│  • Receives debate batch scores from OversightRunner                    │
│  • StatReport: serializes to JSON, prints rich summary table            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT LAYER  (outputs/{experiment_id}/)                               │
│  • results.jsonl              — per-item scores, token counts           │
│  • oversight_results.jsonl    — per-item agent verdicts                 │
│  • oversight_manifest.json    — run config snapshot, seeds, timestamp   │
│  • oversight_stats.json       — StatEngine output for debate batch      │
│  • traces/{query_id}.json     — full A→B→C trace per debate item        │
│  • manifest.json              — run config snapshot (metric runs)       │
│  • cost_ledger_{id}.jsonl     — per-call token + cost accounting        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Verity/
├── eval_engine/
│   ├── cli.py                  ← Typer CLI (verity run / oversight-run / validate-config / list-metrics)
│   ├── config.py               ← Pydantic v2 EvalConfig + YAML loader
│   ├── runner.py               ← Async EvalRunner: semaphore, retry, JSONL streaming
│   ├── cost_tracker.py         ← Per-call token accounting; enforces budget ceiling pre-call
│   ├── sanitizer.py            ← Prompt injection sanitization applied to all Judge inputs
│   ├── statistics.py           ← StatEngine: Wilcoxon, Cohen's d, Pearson, ECE; StatReport serializer
│   ├── schemas.py              ← EvalRecord, MetricResult, AgentVerdict, RunManifest typed schemas
│   ├── registry.py             ← Metric plugin registry (registry.register / registry.get)
│   │
│   ├── metrics/
│   │   ├── base.py             ← BaseMetric ABC (abstract .score() → MetricResult)
│   │   ├── retrieval.py        ← ndcg, recall_at_k, mean_reciprocal_rank
│   │   ├── grounding.py        ← ragas_grounding, ragas_consolidation_delta
│   │   ├── graph_refinement.py ← compression_delta, deduplication_delta, entity_coverage
│   │   ├── safety.py           ← llamaguard_safety (optional dep: transformers, torch)
│   │   ├── adversarial.py      ← single_session_poisoning, query_perturbation
│   │   ├── alignment.py        ← calibration, hallucination_rate, trust_score
│   │   ├── ablation.py         ← per_stage_ablation, threshold_compute_budget
│   │   ├── longitudinal.py     ← multi_session_persistence
│   │   └── scaffolds/          ← consistency, constitutional_eval, model_written_eval,
│   │                               source_reliability, goal_misgeneralization, deceptive_alignment
│   │
│   ├── agents/
│   │   ├── agent_base.py       ← AgentBase ABC: real/dry-run dispatch, trace logging, token accounting
│   │   ├── proposer.py         ← Generates candidate answers from retrieved contexts
│   │   ├── critic.py           ← Challenges grounding; sycophancy pre-screen; flags safety issues
│   │   ├── judge.py            ← Issues Pass / Conditional / Fail verdict
│   │   └── debate_round.py     ← Orchestrates Proposer → Critic → Judge; supports dry_run
│   │
│   └── orchestration/
│       ├── oversight_runner.py ← OversightRunner: dataset-level debate execution wired to CostTracker + StatEngine
│       ├── celery_tasks.py     ← Celery task definitions; dispatches debate rounds to Redis queue
│       └── sync_fallback.py    ← In-process queue (no Redis required)
│
├── configs/
│   └── consolidation_eval_example.yaml   ← Reference config for DreamRAG consolidation eval
│
├── examples/
│   ├── mock_debate/            ← Zero-cost dry_run demo (no API key required)
│   ├── ragas_grounding/        ← RAGAS faithfulness + answer relevance baseline
│   ├── poisoning_test/         ← single_session_poisoning adversarial evaluation
│   ├── ablation_test/          ← per_stage_ablation across consolidation phases
│   └── budget_sweep/           ← Threshold/compute budget sweep across concurrency levels
│
├── tests/
│   ├── test_eval_engine.py         ← Core metrics, sanitizer, budget tracker
│   ├── test_phase2_metrics.py      ← Ablation, poisoning, perturbation metrics
│   ├── test_phase3_pipeline.py     ← Agent pipeline, debate round, orchestration sync
│   ├── test_oversight_runner.py    ← OversightRunner, OversightRunResult, CLI
│   ├── test_scaffolds.py           ← Scaffold registry and interface contracts
│   ├── test_statistics.py          ← StatEngine edge cases
│   └── test_tier1_metrics.py       ← Calibration, hallucination, trust score, persistence
│
├── .github/workflows/ci.yml   ← CI: pytest, ruff, mypy, build check
├── docker-compose.yml          ← Redis + Celery worker + Flower monitor
├── Dockerfile.worker           ← Worker container image
├── pyproject.toml              ← Build system, deps, dev tools, pytest config
└── .env.example                ← Environment variable template
```

---

## 4. Component Deep-Dives

### 4.1 CLI → Config → Runner

The CLI (`cli.py`) is a pure dispatch layer. It parses input, instantiates `EvalConfig`, and routes to either `EvalRunner` (metric evaluation via `verity run`) or `OversightRunner` (debate pipeline via `verity oversight-run`). No business logic lives in `cli.py`.

`EvalConfig` (Pydantic v2) validates all parameters at instantiation time. The `metrics` field accepts an empty list — oversight runs do not require metrics. Fields with constraints (e.g., `concurrency: int = Field(ge=1, le=50)`, `budget: float = Field(gt=0.0)`) fail loudly before a single API call is made.

`EvalRunner` uses `asyncio.Semaphore(config.concurrency)` to bound parallel API calls. Each item is retried up to 3× with exponential backoff on `RateLimitError`. Items are streamed to `results.jsonl` as they complete.

### 4.2 OversightRunner

`OversightRunner` is the dataset-level wrapper around `DebateRound`, introduced in v0.2.1. It closes the wiring gap between the debate pipeline and the rest of the platform:

- Accepts a dataset and `EvalConfig`
- Dispatches each item through `DebateRound` with async semaphore concurrency
- Feeds per-call token counts into the main `CostTracker` — `BudgetExceededError` applies to oversight runs
- Collects `DebateResult` objects and passes safety/accuracy score arrays to `StatEngine`
- Writes `oversight_results.jsonl`, `oversight_manifest.json`, `oversight_stats.json`, and per-debate trace JSONs

```python
# Dry-run (zero cost)
verity oversight-run --dataset datasets/eval_set.json --dry-run

# Live run with budget ceiling
verity oversight-run --dataset datasets/eval_set.json --budget 20.00

# Distributed
verity oversight-run --dataset datasets/eval_set.json --celery
```

### 4.3 Multi-Agent Oversight Pipeline

The Proposer → Critic → Judge pipeline is the system's principal oversight mechanism and its primary defense against reward hacking.

- **Proposer**: Given a query and retrieved contexts, generates a candidate answer. No network access, no tool calls. Output is a structured typed object.
- **Critic**: Receives the query, contexts, and Proposer output. Runs a sycophancy pre-screen before grounding challenge. Flags potential safety issues. Output includes a `reward_hacking_suspected` boolean.
- **Judge**: Receives all prior context plus the Critic's critique. Issues a verdict (`Pass`, `Conditional`, `Fail`) and a `final_safety_score`. The verdict gates whether the item's scores are included in the statistical report.

This structure enforces the OWASP LLM08 (Excessive Agency) mitigation: the Proposer has no ability to trigger downstream actions. Its output must survive Critic review and Judge approval before it influences any scored result.

**Dry-run mode** (`DebateRound(dry_run=True)`) stubs all three agent calls with deterministic fixture responses, enabling zero-cost CI testing and local demos without API keys.

### 4.4 Sanitizer

`sanitizer.py` applies prompt injection defenses to any text passed to the Judge agent. The sanitizer strips common injection patterns before Judge submission. This is not a complete defense — see `SECURITY.md` for the full threat model.

### 4.5 Statistical Engine

`StatEngine` computes all statistics after metric scoring is complete, operating only on arrays of floats. In v0.2.1, it also receives debate batch scores from `OversightRunner` for safety/accuracy analysis.

Edge case handling is explicit:
- **Cohen's d with N < 10**: logs warning and returns `None`
- **Uniform distribution (std=0)**: logs warning and returns `None`
- **Wilcoxon with tied ranks**: uses `method='approx'` with tie-correction warning

### 4.6 Cost Tracker

`CostTracker` maintains a running token ledger across all API calls. In v0.2.1, it is wired into the debate pipeline — per-agent token counts from `AgentBase` feed the main ledger. Before each call, it checks whether the projected cost would exceed `config.budget`. If yes, the call is aborted with `BudgetExceededError`.

### 4.7 Distributed Mode (Redis + Celery)

When `docker compose up -d` is running, both `EvalRunner` and `OversightRunner` detect the Redis URL and route through `celery_tasks.py`. The sync fallback provides identical semantics without Redis, using an in-process queue. This is the default path for `--dry-run` and local use.

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
| Ollama (local) | `pip install -e ".[ollama-backend]"` | None (local endpoint) |

Each agent instantiates its LLM client via a provider factory keyed on `config.model`. The `docker-compose.yml` exposes `PROPOSER_MODEL`, `CRITIC_MODEL`, and `JUDGE_MODEL` as separate environment variables for heterogeneous provider configurations.

---

## 7. Known Architectural Limitations

- **No GUI.** Verity is CLI + SDK only.
- **LlamaGuard requires local GPU.** The `llamaguard_safety` metric uses `transformers` + `torch` and requires 16GB+ VRAM. Excluded from CI.
- **Distributed mode not load-tested.** Celery + Redis is implemented and functional; large-scale concurrency has not been benchmarked.
- **Judge sanitizer is partial.** Prompt injection resistance for the Judge is a known open problem. Current sanitizer covers common patterns.
- **mypy union-attr on Anthropic SDK.** The Anthropic content block union type causes mypy to flag `.text` access. Suppressed with `# type: ignore[union-attr]`; runtime behavior is correct.

---

*Verity is reproducible open-source research infrastructure for AI assurance, RAG evaluation, and scalable oversight. It is not a finished enterprise product.*

*© 2026 Aurelian Security — MIT License*
