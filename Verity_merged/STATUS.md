# STATUS.md — Verity Implementation Maturity

**Aurelian Security | Verity v0.5.1 | Updated: June 2026**

> This document separates what is implemented and tested from what is experimental or planned. Read this before citing Verity capabilities in a paper or presenting it to a research audience.

---

## Current Version: v0.5.0

Verity is **pre-1.0 research infrastructure**. The API is not stable. Breaking changes between minor versions are expected and will be documented in `CHANGELOG.md`.

---

## Feature Status Matrix

### Core Infrastructure

| Component | Status | Notes |
|---|---|---|
| CLI (`verity run`, `validate-config`, `list-metrics`) | ✅ Implemented | Stable interface |
| `verity oversight-run` CLI command | ✅ Implemented | Full flag set: `--dry-run`, `--budget`, `--celery`, model flags |
| `verity compare` CLI command | ✅ Implemented | Diffs two run manifests; saves comparison JSON |
| `verity manifest` CLI command | ✅ Implemented | Dataset SHA-256 hash + schema without running eval |
| Pydantic v2 EvalConfig (YAML + flags) | ✅ Implemented | Full validation at parse time; `seed` and `track` fields added |
| Async EvalRunner (semaphore + retry) | ✅ Implemented | Exponential backoff on RateLimitError; seed control integrated |
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
| **Reproducibility bundle** | ✅ Implemented | Seed, git hash, Python version, all installed packages per run |
| **Dataset manifest + versioning** | ✅ Implemented | SHA-256 content hash, schema, lineage chain |
| **Run comparison** | ✅ Implemented | Score deltas, verdict diffs, dataset identity verification |
| **Global seed control** | ✅ Implemented | `--seed` flag wired into all random sampling and perturbation |

---

### Metric Registry

#### Implemented (25 metrics)

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
| **`consistency`** | **Consistency / stability** | **Unit tested (Tier 2)** |
| **`constitutional_eval`** | **Alignment rubric** | **Unit tested, dry-run (Tier 2)** |
| **`model_written_eval`** | **Policy rubric** | **Unit tested, dry-run (Tier 2)** |
| **`source_reliability`** | **Source provenance** | **Unit tested (Tier 2)** |
| **`bm25`** | **Retrieval ranking** | **Unit tested** |
| **`rrf`** | **Retrieval ranking** | **Unit tested** |
| **`mmr`** | **Retrieval ranking** | **Unit tested** |
| **`ltr_heuristic`** | **Retrieval ranking** | **Unit tested** |

#### Algorithm Expansion Registry — 47 metrics (v0.5.0)

47 additional metrics in `eval_engine/metrics/algorithm_expansion.py` spanning alignment, safety, and adversarial security. All registered in `MetricsRegistry` and addressable via `verity run`.

| Domain | Count | Status |
|--------|-------|--------|
| Alignment | 15 | 9 heuristic, 6 scaffold |
| Safety | 17 | 13 heuristic, 4 scaffold |
| Security / Adversarial | 15 | 9 heuristic, 6 scaffold |
| **Total** | **47** | **31 heuristic, 16 scaffold** |

Heuristic implementations are deterministic, dependency-free, and fully tested. Scaffold implementations follow the standard scaffold contract (`status: "scaffold"`, zero score) and require external models, GPU, or white-box access. **Algorithms 48–50 are not included in this release.**

#### Gap-Closure Conformance Pass — 43 metrics

43 additional metrics from two staging packages (`verity_gap_closure`, `verity_gap_algorithms`), rebuilt against the live `BaseMetric`/`HeuristicMetric` interface (not the reconstructed interface the staging packages were originally built against). All heuristic — no scaffolds in this batch. Registered directly into `_BUILTIN_REGISTRY` (not yet promoted to `TestName` enum entries, though `MetricConfig.name: TestName | str` means they're reachable via `MetricConfig`/CLI as plain strings already).

| Domain | Count | Status |
|--------|-------|--------|
| Corrigibility | 4 | 4 heuristic |
| Goal misgeneralization / spec gaming | 4 | 4 heuristic |
| Alignment faking | 4 | 4 heuristic |
| Sandbagging | 5 | 5 heuristic |
| Scalable oversight | 4 | 4 heuristic |
| LatentIDS cross-architecture / probe robustness | 3 | 3 heuristic |
| OWASP / agent / application security | 13 | 13 heuristic |
| Judging / provenance infrastructure | 6 | 6 heuristic |
| **Total** | **43** | **43 heuristic, 0 scaffold** |

`probe_distribution_shift_robustness` and `cross_architecture_probe_validation` share a single rank-based AUROC core (`_probe_robustness_core.py`) rather than each reimplementing the same math — a coexistence refactor verified against both metrics' original independent implementations via regression tests.

Input shape note: unlike most of the 47-algorithm expansion registry, this batch operates almost entirely on structured episode/trial/judge-vote/probe-score/manifest data passed via `**kwargs`, not on RAG `(question, contexts, answer, ground_truth)` content. See `gap_closure_expansion.py`/`gap_algorithms_expansion.py` module docstrings.


#### Scaffolded — Tier 3 (2 metrics, frontier research prerequisites)

These metrics have `BaseMetric`-compliant interfaces registered in the registry. Their `.score()` methods return placeholder `MetricResult` objects with `status: "scaffold"` in metadata.

| Metric | Target Tier | Blocker |
|---|---|---|
| `goal_misgeneralization` | Tier 3 | Papers 1–2 published; OOD test set designed and validated |
| `deceptive_alignment` | Tier 3 | Phase 3 RH detection validated on real data; LatentIDS available |

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
| Per-debate trace JSON output | ✅ Implemented | Full A→B→C trace per item in `debates/` subdirectory |
| Cost tracking wired into pipeline | ✅ Implemented | Per-agent token counts feed main CostTracker |
| StatEngine integration for debate batches | ✅ Implemented | Safety/accuracy score analysis across runs |
| Debate loop stress testing | ⚠️ Not formally tested | Long recursive loops not benchmarked |

---

### Statistical Engine

| Capability | Status | Notes |
|---|---|---|
| Wilcoxon rank-sum (pre/post delta) | ✅ Implemented | Tie-corrected with warning |
| Mann-Whitney U (independent groups) | ✅ Implemented | Cliff's delta effect size included |
| Paired t-test (parametric) | ✅ Implemented | Recommended only after Shapiro confirms normality |
| Cohen's d (effect size) | ✅ Implemented | Guards small-N and zero-variance; returns None with warning |
| Cliff's delta (non-parametric effect size) | ✅ Implemented | Recommended for adversarial score distributions |
| Pearson correlation | ✅ Implemented | |
| Spearman correlation | ✅ Implemented | |
| Shapiro-Wilk normality test | ✅ Implemented | |
| Descriptive statistics | ✅ Implemented | mean, median, std, IQR, min, max |
| pre_post_bundle() (7-test suite) | ✅ Implemented | One call for full results table row |
| StatReport (JSON + rich summary) | ✅ Implemented | |
| Debate batch statistical analysis | ✅ Implemented | Safety/accuracy score analysis via OversightRunner |
| Bootstrap confidence intervals | 🔲 Planned | |
| Multiple comparison correction | 🔲 Planned | Bonferroni / Benjamini-Hochberg |

---

### Tier 2 Metrics (v0.4.0)

| Component | Status | Notes |
|---|---|---|
| `consistency` — jaccard mode | ✅ Implemented | Token-overlap similarity; no dependencies |
| `consistency` — embedding mode | ✅ Implemented | sentence-transformers cosine similarity; `pip install sentence-transformers` |
| `consistency` — accumulator mode | ✅ Implemented | `score()` accumulates, `compute_consistency_report()` aggregates |
| `consistency` — Consolidation Stability Index | ✅ Implemented | Pre/post drift ratio; requires pre_consolidation_answers |
| `constitutional_eval` — 5 dimensions | ✅ Implemented | Harmlessness, honesty, transparency, non-manipulation, privacy |
| `constitutional_eval` — dry-run mode | ✅ Implemented | Deterministic mock scores, zero API cost |
| `constitutional_eval` — weighted composite | ✅ Implemented | Equal weighting default; customizable |
| `model_written_eval` — binary criteria | ✅ Implemented | PASS/FAIL → 1.0/0.0 |
| `model_written_eval` — scored criteria | ✅ Implemented | 1-5 and 1-10 scales normalized to 0.0–1.0 |
| `model_written_eval` — required criteria | ✅ Implemented | Failure forces composite to 0.0 |
| `model_written_eval` — default_governance_rubric() | ✅ Implemented | 4-criterion governance rubric |
| `model_written_eval` — research_paper_rubric() | ✅ Implemented | 4-criterion research quality rubric |
| `model_written_eval` — dry-run mode | ✅ Implemented | Mock scores, zero API cost |
| `source_reliability` — authority scoring | ✅ Implemented | Domain taxonomy (arxiv=0.85, nature=0.95, wikipedia=0.65, etc.) |
| `source_reliability` — citation scoring | ✅ Implemented | Logarithmic scale; 0 citations → 0.0 |
| `source_reliability` — recency scoring | ✅ Implemented | Exponential decay; configurable half-life |
| `source_reliability` — consistency heuristic | ✅ Implemented | Authority variance proxy; no LLM cost |
| `source_reliability` — LLM consistency check | ✅ Implemented | Optional (`check_consistency=True`); adds API cost |
| `source_reliability` — score_from_result() | ✅ Implemented | Typed interface via RetrievalResult |
| `schemas.py` — SourceMetadata | ✅ Implemented | Full dataclass with all source provenance fields |
| `schemas.py` — RetrievalResult.source_metadata | ✅ Implemented | Optional field; backward-compatible |

---

### Reproducibility Infrastructure (v0.3.0)

| Component | Status | Notes |
|---|---|---|
| `ReproducibilityBundle` | ✅ Implemented | Captures seed, git hash, Python version, all installed packages |
| `set_global_seed()` | ✅ Implemented | Seeds Python random + numpy + torch |
| `--seed` flag on `verity run` | ✅ Implemented | Flows through EvalConfig into all random sampling |
| `--track` flag on `verity run` | ✅ Implemented | Writes reproducibility.json + dataset_manifest.json |
| `DatasetManifest` (SHA-256 + lineage) | ✅ Implemented | Content-addresses datasets; records transformation chain |
| `verity manifest` CLI command | ✅ Implemented | Standalone dataset inspection without running eval |
| `verity compare` CLI command | ✅ Implemented | Diffs two run manifests |
| `compare_run_manifests()` SDK function | ✅ Implemented | Returns structured comparison dict |
| Context redaction in outputs (`--redact-contexts`) | 🔲 Planned | See SECURITY.md |

---

### Distributed / Orchestration

| Component | Status | Notes |
|---|---|---|
| Celery + Redis distributed mode | ✅ Implemented | Functional; not load-tested |
| `verity oversight-run --celery` | ✅ Implemented | Dispatches debate rounds via Celery queue |
| Sync fallback (no Redis required) | ✅ Implemented | Default for `--dry-run` and local use |
| Flower monitor UI | ✅ Implemented | `docker compose --profile monitoring up` |
| Worker auto-scaling | ⚠️ Manual | `docker compose up --scale worker=N` |
| Distributed mode at scale (>50 concurrent workers) | 🔲 Not validated | See Known Limitations |

---

### CI/CD & Developer Tooling

| Item | Status | Notes |
|---|---|---|
| `pytest` test suite (253 tests) | ✅ Passing | Unit + integration; 253/253 green |
| `ruff` linting | ✅ Passing | |
| `mypy` type checking | ⚠️ Non-blocking | Known union-attr issues with Anthropic SDK types |

---

## Known Limitations (Non-Negotiable Honesty)

These are current facts about Verity, not aspirational caveats:

- **Not validated at large scale.** Celery + Redis architecture is implemented and functional; concurrent worker performance under >50 parallel debate rounds has not been benchmarked.
- **Empirical validation pending.** The `ragas_consolidation_delta` metric and consolidation evaluation infrastructure are designed for consolidation-based RAG architectures. Empirical validation against published benchmarks is in progress.
- **LlamaGuard requires 16GB+ VRAM.** The `llamaguard_safety` metric is excluded from standard CI and is not available in CPU-only environments.
- **No GUI.** Verity is CLI + SDK only. There is no web dashboard.
- **`multi_session_persistence` has limited test coverage.** Longitudinal evaluation across multiple sessions requires a persistent dataset fixture not yet fully implemented in the test suite.
- **Prompt injection sanitizer is partial.** The current sanitizer covers known pattern-based attacks. Semantically coherent adversarial injections designed to evade pattern matching are not addressed.
- **mypy union-attr errors from Anthropic SDK.** The Anthropic SDK's content block union type causes mypy to flag `.text` access as unsafe. These are suppressed with `# type: ignore[union-attr]`; the runtime behavior is correct.
- **No formal security audit.** Verity has not undergone a professional penetration test or formal code security review.
- **Redis has no authentication by default.** The default `docker-compose.yml` does not set `requirepass`. Add Redis authentication before any non-localhost deployment.

---

## What "Experimental" Means Here

A metric or component marked **Experimental** has:
- A functioning implementation
- No guarantee of API stability
- Insufficient test coverage to assert correctness under edge cases
- No published validation against ground-truth datasets

A component marked **Scaffolded** has:
- A registered interface that will not break your import
- A placeholder `.score()` that returns zero with `status: "scaffold"` metadata
- A clear blocker documented above

Do not include scaffolded metrics in published benchmark results.

---

*Verity is reproducible open-source research infrastructure for RAG evaluation and adversarial robustness testing. It is not a finished enterprise product.*

*© 2026 Aurelian Security — MIT License*
