# STATUS.md — Verity Implementation Maturity

**Aurelian Security | Verity v0.2.1 | Updated: June 2026**

> This document separates what is implemented and tested from what is experimental or planned. Read this before citing Verity capabilities in a paper or presenting it to a research audience.

---

## Current Version: v0.2.1

Verity is **pre-1.0 research infrastructure**. The API is not stable. Breaking changes between minor versions are expected and will be documented in `CHANGELOG.md`.

---

## Feature Status Matrix

### Core Infrastructure

| Component | Status | Notes |
|---|---|---|
| CLI (`verity run`, `validate-config`, `list-metrics`) | ✅ Implemented | Stable interface |
| `verity oversight-run` CLI command | ✅ Implemented | Full flag set: `--dry-run`, `--budget`, `--celery`, model flags |
| Pydantic v2 EvalConfig (YAML + flags) | ✅ Implemented | Full validation at parse time |
| Async EvalRunner (semaphore + retry) | ✅ Implemented | Exponential backoff on RateLimitError |
| OversightRunner (dataset-level debate execution) | ✅ Implemented | Wraps DebateRound; wired to CostTracker + StatEngine |
| JSONL streaming output | ✅ Implemented | Per-item, not batched at end |
| Oversight output suite | ✅ Implemented | `oversight_results.jsonl`, `oversight_manifest.json`, `oversight_stats.json`, per-debate trace JSON |
| Cost tracking + budget ceiling | ✅ Implemented | BudgetExceededError on breach; wired into debate pipeline |
| Cost tracking wired into debate pipeline | ✅ Implemented | Per-agent token accounting feeds main CostTracker |
| StatEngine integration for debate results | ✅ Implemented | Safety/accuracy score analysis across debate batches |
| Prompt injection sanitizer | ✅ Implemented | Pattern-based; see SECURITY.md |
| Run manifest (config snapshot + timestamp) | ✅ Implemented | Written to `manifest.json` per run |
| Mock / dry-run mode | ✅ Implemented | `DebateRound(dry_run=True)`; zero API cost |
| Metric plugin registry | ✅ Implemented | `registry.register()` for custom metrics |

---

### Metric Registry

#### Implemented (17 metrics)

| Metric | Category | Test Coverage |
|---|---|---|
| `ndcg` | Retrieval effectiveness | Unit tested |
| `recall_at_k` | Retrieval effectiveness | Unit tested |
| `mean_reciprocal_rank` | Retrieval effectiveness | Unit tested |
| `ragas_grounding` | Retrieval grounding | Integration tested |
| `ragas_consolidation_delta` | Consolidation evaluation | Integration tested |
| `compression_delta` | Graph refinement | Unit tested |
| `deduplication_delta` | Graph refinement | Unit tested |
| `entity_coverage` | Graph refinement | Unit tested |
| `llamaguard_safety` | Safety | Tested (requires GPU) |
| `per_stage_ablation` | Ablation analysis | Integration tested |
| `threshold_compute_budget` | Compute efficiency | Unit tested |
| `single_session_poisoning` | Adversarial robustness | Unit tested |
| `query_perturbation` | Adversarial robustness | Unit tested |
| `calibration` | Trust / alignment | Unit tested |
| `hallucination_rate` | Trust / alignment | Unit tested |
| `trust_score` | Trust / alignment | Unit tested |
| `multi_session_persistence` | Longitudinal evaluation | ⚠️ Limited testing |

#### Scaffolded — Interface Locked, Implementation Deferred (6 metrics)

These metrics have `BaseMetric`-compliant interfaces registered in the registry, but their `.score()` methods raise `NotImplementedError`. They are listed in `verity list-metrics` with a `[scaffold]` tag.

| Metric | Target Tier | Blocker |
|---|---|---|
| `consistency` | Tier 2 | Requires multi-run dataset design |
| `constitutional_eval` | Tier 2 | Requires Constitutional AI prompt library |
| `model_written_eval` | Tier 2 | Requires eval dataset generation pipeline |
| `source_reliability` | Tier 2 | Requires external source credibility DB |
| `goal_misgeneralization` | Tier 3 | Active research; no consensus metric |
| `deceptive_alignment` | Tier 3 | Active research; no consensus metric |

---

### Multi-Agent Oversight Pipeline

| Component | Status | Notes |
|---|---|---|
| Proposer agent | ✅ Implemented | Text generation only; no tool calls |
| Critic agent | ✅ Implemented | Grounding challenge + safety flag; sycophancy pre-screen |
| Judge agent | ✅ Implemented | Pass / Conditional / Fail verdict |
| Dry-run fixtures | ✅ Implemented | Deterministic; usable in CI without API keys |
| Reward hacking detection flag | ✅ Implemented | `reward_hacking_confirmed` in verdict |
| OversightRunner (dataset-level execution) | ✅ Implemented | Async with semaphore; feeds CostTracker + StatEngine |
| Per-debate trace JSON output | ✅ Implemented | Full A→B→C trace per item |
| Cost tracking wired into pipeline | ✅ Implemented | Per-agent token counts feed main CostTracker |
| StatEngine integration for debate batches | ✅ Implemented | Safety/accuracy score analysis across runs |
| Debate loop stress testing | ⚠️ Not formally tested | Long recursive loops not benchmarked |

---

### Statistical Engine

| Capability | Status | Notes |
|---|---|---|
| Wilcoxon rank-sum (pre/post delta) | ✅ Implemented | Tie-corrected with warning |
| Cohen's d (effect size) | ✅ Implemented | Guards small-N and zero-variance; returns None with warning |
| Pearson correlation | ✅ Implemented | |
| ECE (calibration error) | ✅ Implemented | |
| StatReport (JSON + rich summary) | ✅ Implemented | |
| Debate batch statistical analysis | ✅ Implemented | Safety/accuracy score analysis via OversightRunner |
| Bootstrap confidence intervals | 🔲 Planned v0.3 | |
| Multiple comparison correction (Bonferroni / BH) | 🔲 Planned v0.3 | |

---

### Distributed / Orchestration

| Component | Status | Notes |
|---|---|---|
| Celery + Redis distributed mode | ✅ Implemented | Functional; not load-tested |
| `verity oversight-run --celery` | ✅ Implemented | Dispatches debate rounds via Celery queue |
| Sync fallback (no Redis required) | ✅ Implemented | Default for `--dry-run` and local use |
| Flower monitor UI | ✅ Implemented | `docker compose --profile monitoring up` |
| EvalRunner ↔ OversightRunner integration | ✅ Implemented | Single config can run metrics + debate pipeline |
| Worker auto-scaling | ⚠️ Manual | `docker compose up --scale worker=N` |
| Distributed mode at scale (>50 concurrent workers) | 🔲 Not validated | See Known Limitations |

---

### CI/CD & Developer Tooling

| Item | Status | Notes |
|---|---|---|
| `pytest` test suite (191 tests) | ✅ Passing | Unit + integration; 191/191 green |
| `ruff` linting | ✅ Passing | CI-enforced |
| `mypy` type checking | ⚠️ Non-blocking | Runs in CI; known union-attr issues with Anthropic SDK types |
| GitHub Actions CI workflow | ✅ Implemented | `.github/workflows/ci.yml` |
| Build check (`pip install -e .`) | ✅ Implemented | Part of CI |
| Coverage reporting | 🔲 Planned v0.3 | |

---

## Known Limitations (Non-Negotiable Honesty)

These are current facts about Verity, not aspirational caveats:

- **Not validated at large scale.** Celery + Redis architecture is implemented and functional; concurrent worker performance under >50 parallel debate rounds has not been benchmarked. Do not treat Celery throughput claims as empirically verified.
- **DreamRAG empirical results pending.** The `ragas_consolidation_delta` metric and the consolidation evaluation infrastructure were designed in conjunction with DreamRAG research (Kwaai AI Lab). Empirical validation of consolidation-phase retrieval improvement is in progress and has not been published.
- **LlamaGuard requires 16GB+ VRAM.** The `llamaguard_safety` metric is excluded from standard CI and is not available in CPU-only environments. The base install does not include `transformers` or `torch`.
- **No GUI.** Verity is CLI + SDK only. There is no web dashboard.
- **`multi_session_persistence` has limited test coverage.** Longitudinal evaluation across multiple sessions requires a persistent dataset fixture that is not yet fully implemented in the test suite.
- **Prompt injection sanitizer is partial.** The current sanitizer covers known pattern-based attacks. Semantically coherent adversarial injections designed to evade pattern matching are not addressed.
- **mypy union-attr errors from Anthropic SDK.** The Anthropic SDK's content block union type (`TextBlock | ThinkingBlock | ...`) causes mypy to flag `.text` access as unsafe. These are suppressed with `# type: ignore[union-attr]` at call sites; the runtime behavior is correct.
- **No formal security audit.** Verity has not undergone a professional penetration test or formal code security review.

---

## What "Experimental" Means Here

A metric or component marked **Experimental** has:
- A functioning implementation
- No guarantee of API stability
- Insufficient test coverage to assert correctness under edge cases
- No published validation against ground-truth datasets

A component marked **Scaffolded** has:
- A registered interface that will not break your import
- A `NotImplementedError` on `.score()` invocation
- A clear blocker documented above

Do not include scaffolded metrics in published benchmark results.

---

*Verity is reproducible open-source research infrastructure for AI assurance, RAG evaluation, and scalable oversight. It is not a finished enterprise product.*

*© 2026 Aurelian Security — MIT License*
