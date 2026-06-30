# Verity Evaluation Engine

**Verity** is an open-source RAG evaluation SDK built by [Aurelian Security](https://github.com/Aurelian-Security).

Verity provides a reproducible evaluation harness for RAG architectures — CLI-driven, async-batched, with pluggable metrics, multi-agent oversight, statistical analysis, experiment tracking, and per-run cost accounting. Designed to support publication-grade reproducibility and adversarial robustness testing.

> *Trust, measured.*

[![Tests](https://img.shields.io/badge/tests-253%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Version](https://img.shields.io/badge/version-0.5.0-blue)]()

> **Pre-1.0 research infrastructure.** The API is not stable. See [STATUS.md](STATUS.md) for implementation maturity, known limitations, and what is safe to cite in a paper.

---

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [SDK Usage](#sdk-usage)
- [Metric Registry](#metric-registry)
- [Multi-Agent Oversight Pipeline](#multi-agent-oversight-pipeline)
- [Statistical Analysis](#statistical-analysis)
- [Reproducible Experiments](#reproducible-experiments)
- [Dataset Versioning](#dataset-versioning)
- [Run Comparison](#run-comparison)
- [Distributed Execution](#distributed-execution)
- [Architecture](#architecture)
- [Configuration Reference](#configuration-reference)
- [Output Structure](#output-structure)
- [Environment Variables](#environment-variables)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Install

```bash
pip install -e .

# With LlamaGuard safety classifier (requires GPU / 16GB+ RAM):
pip install -e ".[safety]"

# With Anthropic backend for RAGAS:
pip install -e ".[anthropic-backend]"

# Dev tools (pytest, ruff, mypy):
pip install -e ".[dev]"
```

Verify installation:

```bash
verity --help
```

---

## Quick Start

### 1. Create a dataset

```json
[
  {
    "question": "What does the consolidation phase improve?",
    "contexts": ["The consolidation phase restructures the knowledge graph by pruning low-weight edges."],
    "answer": "Consolidation improves retrieval precision by removing redundant connections.",
    "ground_truth": "The consolidation phase performs graph downscaling to improve retrieval quality."
  }
]
```

Save as `datasets/eval_set.json`.

### 2. Run an evaluation

```bash
verity run \
  --dataset datasets/eval_set.json \
  --metrics ndcg,ragas_grounding \
  --model claude-sonnet-4-6 \
  --budget 2.00
```

### 3. Run the oversight pipeline (no API cost)

```bash
verity oversight-run --dataset datasets/eval_set.json --dry-run
```

### 4. Check what was produced

```
outputs/{experiment_id}/
  results.jsonl
  manifest.json
  cost_ledger_{id}.jsonl
```

---

## CLI Reference

### `verity run`

Run a metric evaluation against a dataset.

```bash
verity run --config configs/consolidation_eval_example.yaml
```

```bash
verity run \
  --dataset datasets/eval_set.json \
  --metrics ndcg,ragas_consolidation_delta,llamaguard_safety \
  --model claude-sonnet-4-6 \
  --architecture consolidation \
  --budget 5.00 \
  --concurrency 10 \
  --seed 42 \
  --track
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Path to YAML EvalConfig (overrides all flags) |
| `--dataset` | — | Path to dataset (JSON/JSONL/CSV) |
| `--metrics` | — | Comma-separated metric names |
| `--model` | `claude-sonnet-4-6` | LLM judge model |
| `--architecture` | `centralized` | `centralized` \| `decentralized` \| `consolidation` |
| `--budget` | `5.00` | Max USD spend before halt |
| `--concurrency` | `10` | Max concurrent async queries |
| `--seed` | `42` | Random seed for sampling and perturbation |
| `--track` | `False` | Write reproducibility bundle + dataset manifest |
| `--output-dir` | `outputs/` | Results directory |
| `--verbose` | `False` | Debug logging |

---

### `verity oversight-run`

Run a dataset through the multi-agent oversight pipeline (Proposer → Critic → Judge).

```bash
verity oversight-run --dataset datasets/eval_set.json --dry-run
```

```bash
verity oversight-run \
  --dataset datasets/eval_set.json \
  --proposer-model claude-haiku-4-5 \
  --critic-model claude-sonnet-4-6 \
  --judge-model claude-haiku-4-5 \
  --recall-at-k 5 \
  --budget 10.00 \
  --concurrency 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | Path to YAML EvalConfig |
| `--dataset` | — | Path to dataset |
| `--dry-run` | `False` | Mock responses, zero API cost |
| `--proposer-model` | `claude-haiku-4-5` | Agent A model |
| `--critic-model` | `claude-sonnet-4-6` | Agent B model (reward hacking detection) |
| `--judge-model` | `claude-haiku-4-5` | Agent C model |
| `--recall-at-k` | `5` | Context chunks critic verifies against |
| `--budget` | `10.00` | Max USD spend before halt |
| `--concurrency` | `4` | Max concurrent debate rounds |
| `--celery` | `False` | Dispatch to Celery workers (requires Redis) |

---

### `verity compare`

Diff two run manifests side by side.

```bash
verity compare --run-a consolidation_eval_run1 --run-b consolidation_eval_run2
```

Shows: metric score deltas, verdict distribution changes, dataset identity verification, git hash comparison. Saves `outputs/compare_A_vs_B.json`.

---

### `verity manifest`

Generate a dataset manifest (content hash + schema) without running an eval.

```bash
verity manifest datasets/eval_set.json --version-tag v1.0 --output manifests/
```

---

### `verity validate-config`

Validate a YAML config file without running.

```bash
verity validate-config configs/consolidation_eval_example.yaml
```

---

### `verity list-metrics`

List all registered metrics with implementation status.

```bash
verity list-metrics
```

---

## SDK Usage

### Metric evaluation

```python
import asyncio
from eval_engine.config import EvalConfig
from eval_engine.runner import EvalRunner

config = EvalConfig.from_yaml("configs/consolidation_eval_example.yaml")
runner = EvalRunner(config)

dataset = [
    {
        "question": "What does the consolidation phase improve?",
        "contexts": ["The consolidation phase restructures the knowledge graph..."],
        "answer": "Consolidation improves retrieval precision.",
        "ground_truth": "The consolidation phase performs graph downscaling.",
        # For consolidation delta:
        "pre_contexts": ["Pre-consolidation chunk..."],
        "pre_answer": "Pre-consolidation answer...",
    }
]

result = asyncio.run(runner.run(dataset))
print(f"NDCG@10:              {result.mean_score('ndcg'):.4f}")
print(f"Consolidation delta:  {result.mean_score('ragas_consolidation_delta'):.4f}")
print(f"Safety score:         {result.mean_score('llamaguard_safety'):.4f}")
```

### Oversight pipeline

```python
import asyncio
from eval_engine.config import EvalConfig, DatasetConfig
from eval_engine.orchestration import OversightRunner

config = EvalConfig(
    experiment_id="oversight_run_1",
    architecture="consolidation",
    model="claude-sonnet-4-6",
    dataset=DatasetConfig(path="datasets/eval_set.json"),
)

runner = OversightRunner(config, dry_run=False)
result = asyncio.run(runner.run(dataset))

print(f"Reward hacking rate:  {result.reward_hacking_rate:.1%}")
print(f"Mean safety score:    {result.mean_safety_score():.4f}")
print(f"Verdict distribution: {result.verdict_distribution}")
```

### Single debate round

```python
from eval_engine.orchestration import DebateRound

round = DebateRound(dry_run=True)
result = round.run(
    query_id="q1",
    query="What does the consolidation phase improve?",
    contexts=["Context A...", "Context B..."],
)

print(result.verdict)                    # Pass | Conditional | Fail
print(result.final_safety_score)         # 0.0 – 1.0
print(result.final_accuracy_score)       # 0.0 – 1.0
print(result.reward_hacking_confirmed)   # True | False
```

### Custom metric plugin

```python
from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine import registry

class MyCustomMetric(BaseMetric):
    name = "my_metric"

    def score(self, question, contexts, answer, ground_truth=None, **kwargs):
        return MetricResult(metric_name=self.name, score=0.95)

registry.register("my_metric", MyCustomMetric)
```

---

## Metric Registry

Verity's registry contains **74 registered metrics** across four groups.

### Core — Implemented (21)

| Metric | Category | Description |
|--------|----------|-------------|
| `ndcg` | Retrieval | NDCG@K with exponential DCG formula |
| `recall_at_k` | Retrieval | Query-level binary recall |
| `mean_reciprocal_rank` | Retrieval | MRR across benchmark queries |
| `ragas_grounding` | Grounding | RAGAS faithfulness + context precision/recall |
| `ragas_consolidation_delta` | Consolidation | Pre/post RAGAS faithfulness delta |
| `compression_delta` | Graph | Node/edge reduction ratio after consolidation |
| `deduplication_delta` | Graph | Duplicate reduction ratio |
| `entity_coverage` | Graph | Covered / expected entities |
| `llamaguard_safety` | Safety | LlamaGuard-3-8B binary safe/unsafe classification |
| `per_stage_ablation` | Ablation | Per-stage NDCG@K contribution measurement |
| `threshold_compute_budget` | Compute | NDCG@K vs compute budget curve + inflection point |
| `single_session_poisoning` | Adversarial | Attack success rate, LlamaGuard FNR, faithfulness degradation |
| `query_perturbation` | Adversarial | Perturbation Robustness Score across 4 perturbation types |
| `calibration` | Trust | ECE + Brier Score confidence calibration |
| `hallucination_rate` | Trust | Claim-level: unsupported / total claims |
| `trust_score` | Trust | Weighted composite: safety + faithfulness + calibration + hallucination |
| `multi_session_persistence` | Longitudinal | N-cycle forgetting curve, poison/truth persistence rates |
| `consistency` | Stability | Answer drift score across N runs; Consolidation Stability Index |
| `constitutional_eval` | Alignment | 5-dimension LLM rubric: harmlessness, honesty, transparency, non-manipulation, privacy |
| `model_written_eval` | Policy | Caller-defined policy rubrics; binary/scored criteria; required criteria |
| `source_reliability` | Provenance | Authority, citation, recency, and cross-source consistency scoring |

### Retrieval Ranking — Implemented (4)

Classical IR ranking algorithms available as standalone metrics or combined with retrieval effectiveness metrics.

| Metric | Category | Description |
|--------|----------|-------------|
| `bm25` | Retrieval Ranking | BM25 probabilistic ranking; scores contexts against query terms |
| `rrf` | Retrieval Ranking | Reciprocal Rank Fusion; merges multiple rankings into a consensus ranking |
| `mmr` | Retrieval Ranking | Maximal Marginal Relevance; balances relevance against redundancy |
| `ltr_heuristic` | Retrieval Ranking | Learning-to-rank heuristic; weighted query/answer/ground-truth overlap |

**Retrieval effectiveness** (Recall@K, MRR, NDCG) answers: were the right documents retrieved?
**Retrieval ranking** (BM25, RRF, MMR, LTR) answers: were they ranked in the right order and with appropriate diversity?

```yaml
# Example config combining both layers
metrics:
  - name: recall_at_k
  - name: ndcg
  - name: bm25
  - name: rrf
  - name: mmr
  - name: ltr_heuristic
```

### Algorithm Expansion — 47 metrics (`algorithm_expansion.py`)

47 additional metrics spanning alignment, safety, and adversarial security. **31 are deterministic heuristic implementations** (no dependencies, fully tested). **16 are interface-locked scaffolds** requiring external models, GPU, or white-box access — they return zero scores with `status: "scaffold"` metadata until those prerequisites are met.

**Alignment (15):** `constitutional_ai`, `rlhf_reward_model_probing`, `dpo_delta_scoring`, `activation_steering_vector_analysis`, `representation_engineering_probing`, `scalable_oversight_debate`, `process_based_supervision`, `weak_to_strong_generalization_probing`, `mechanistic_interpretability_circuit_detection`, `goodharts_law_metric_stress_testing`, `sycophancy_detection_suite`, `specification_gaming_detection`, `deceptive_alignment_behavioral_testing`, `truthfulness_calibration`, `alignment_tax_measurement`

**Safety (17):** `poisoned_rag_detection`, `prompt_injection_resistance`, `jailbreak_robustness_benchmarking`, `llamaguard_input_output_classification`, `gcg_attack_generation`, `textattack_augmentation_pipeline`, `refusal_consistency_testing`, `multi_turn_safety_degradation_testing`, `hallucination_detection_ragas_factscore`, `toxic_content_classifier_ensemble`, `data_exfiltration_resistance_testing`, `backdoor_trigger_detection`, `membership_inference_attack_testing`, `contextual_integrity_violation_detection`, `semantic_consistency_distribution_shift`, `safe_decoding_integration`, `reward_model_overoptimization_detection`

**Security (15):** `owasp_llm_top_10_compliance_audit`, `mitre_atlas_threat_mapping`, `indirect_prompt_injection_web_content`, `differential_privacy_compliance_testing`, `adversarial_retrieval_ranking_manipulation`, `embedding_inversion_attack_testing`, `supply_chain_integrity_verification`, `adversarial_document_chunking_attacks`, `cross_encoder_reranking_robustness`, `api_rate_limiting_abuse_detection`, `model_extraction_attack_resistance`, `cryptographic_audit_log_integrity`, `semantic_similarity_label_leakage_detection`, `red_team_coverage_matrix`, `adversarial_hyperparameter_search`

### Scaffolded — Tier 3 (frontier research, 2)

| Metric | Research Question |
|--------|------------------|
| `goal_misgeneralization` | Did the system optimize a proxy objective instead of the intended one? |
| `deceptive_alignment` | Does the system behave differently when it detects evaluation? |

> Scaffolded metrics have registered interfaces and return placeholder results. Do not include them in published benchmark results. See [STATUS.md](STATUS.md).

---

## Multi-Agent Oversight Pipeline

Verity's oversight pipeline runs every query through three agents in sequence:

```
Proposer (Agent A) → Critic (Agent B) → Judge (Agent C)
```

**Agent A — Proposer:** Generates a grounded answer from retrieved context. Declares confidence and supporting evidence.

**Agent B — Critic:** Audits the proposal for reward hacking. Runs a rule-based sycophancy pre-screen (zero cost), then an LLM deep audit. Detects: sycophantic openers, confidence inflation, factual drift, metric gaming.

**Agent C — Judge:** Synthesizes the debate. Issues `verdict` (Pass/Conditional/Fail), `final_safety_score`, `final_accuracy_score`, `reward_hacking_confirmed`.

### Dry-run (zero cost)

```bash
verity oversight-run --dataset datasets/eval_set.json --dry-run
```

### Real run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
verity oversight-run \
  --dataset datasets/eval_set.json \
  --proposer-model claude-haiku-4-5 \
  --critic-model claude-sonnet-4-6 \
  --judge-model claude-haiku-4-5 \
  --budget 10.00
```

### Distributed (Redis + Celery)

```bash
cp .env.example .env      # Set ANTHROPIC_API_KEY
docker compose up -d      # Start Redis + workers
verity oversight-run --dataset datasets/eval_set.json --celery
```

Monitor workers at `http://localhost:5555` (Flower UI, requires `--profile monitoring`).

---

## Statistical Analysis

```python
from eval_engine.statistics import StatEngine, StatReport

engine = StatEngine(alpha=0.05)

# Full pre/post analysis bundle (7 tests)
results = engine.pre_post_bundle(pre_scores, post_scores, metric_label="ndcg")
# Returns: descriptive (x2), shapiro normality (x2),
#          wilcoxon, cohens_d, cliffs_delta

report = StatReport(results, experiment_id="consolidation_eval_run1")
report.print_summary()
report.save("outputs/stats_report.json")
```

### Available tests

| Method | Type | Use for |
|--------|------|---------|
| `wilcoxon()` | Significance | Paired pre/post comparisons |
| `mann_whitney()` | Significance | Independent group comparisons |
| `paired_ttest()` | Significance | Large samples with confirmed normality |
| `cohens_d()` | Effect size | Standardized mean difference |
| `cliffs_delta()` | Effect size | Non-parametric, adversarial distributions |
| `pearson()` | Correlation | Linear relationships |
| `spearman()` | Correlation | Monotonic non-linear relationships |
| `shapiro()` | Normality | Gate before parametric vs non-parametric |
| `descriptive()` | Summary | Mean, median, std, IQR, min, max |
| `pre_post_bundle()` | All-in-one | Full results table row (7 tests) |

---

## Reproducible Experiments

Verity captures everything needed to reproduce a run exactly.

### Enable tracking

```bash
verity run --config configs/consolidation_eval_example.yaml --track --seed 42
```

Or in YAML config:

```yaml
seed: 42
track: true
```

### What gets captured

`outputs/{experiment_id}/reproducibility.json`:

```json
{
  "experiment_id": "consolidation_eval_run1",
  "seed": 42,
  "timestamp_iso": "2026-06-25T...",
  "verity_version": "0.3.0",
  "python_version": "3.11.x",
  "git": {
    "hash": "a3f9b2c...",
    "branch": "main",
    "dirty": false
  },
  "installed_packages": { "ragas": "0.1.9", "pydantic": "2.5.0" },
  "config_snapshot": { "..." }
}
```

> **Paper submission note:** Commit all changes before running experiments. If `git.dirty` is `true`, results may not be reproducible from `git.hash` alone.

### Programmatic usage

```python
from eval_engine.reproducibility import ReproducibilityBundle, set_global_seed

set_global_seed(42)   # Seeds Python random + numpy + torch

bundle = ReproducibilityBundle.from_config(config, seed=42)
bundle.capture()
bundle.save(output_dir)
```

---

## Dataset Versioning

Every dataset is content-addressed by SHA-256 hash. The hash is the version ID.

### Generate a manifest

```bash
verity manifest datasets/eval_set.json --version-tag v1.0 --output manifests/
```

### Dataset lineage (poisoning pipeline)

```python
from eval_engine.dataset_manifest import DatasetManifest

manifest = DatasetManifest.from_file("datasets/eval_set.json")

manifest.add_lineage_step(
    "single_session_poisoning",
    output_path="datasets/poisoned_eval_set.json",
    poison_rate=0.1,
    attack_type="factual_substitution",
    seed=42,
)
manifest.save("outputs/poisoning_run1/")
```

`dataset_manifest.json` records the full transformation chain from clean corpus to consolidated state.

---

## Run Comparison

```bash
verity compare --run-a consolidation_eval_run1 --run-b consolidation_eval_run2
```

Output includes metric score deltas, verdict distribution changes, dataset SHA-256 identity verification (warns if datasets differ), git hash comparison, and run statistics. Saves `outputs/compare_A_vs_B.json`.

```python
from eval_engine.comparison import compare_run_manifests

report = compare_run_manifests("run_001", "run_002", output_dir=Path("outputs"))
print(report["score_deltas"])    # {"ndcg": 0.06, "recall_at_k": 0.03}
print(report["same_dataset"])    # True
```

---

## Distributed Execution

```bash
cp .env.example .env                          # Set ANTHROPIC_API_KEY
docker compose up -d                          # Start Redis + workers
docker compose up -d --scale worker=4         # Scale workers
docker compose --profile monitoring up -d     # Flower UI at :5555

verity oversight-run --dataset datasets/eval_set.json --celery --concurrency 4
```

> **Security note:** The default Redis configuration has no authentication. Add `requirepass` and set `REDIS_URL=redis://:password@redis:6379/0` before any non-localhost deployment. See [SECURITY.md](SECURITY.md).

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, data flow diagrams, and component deep-dives.

```
eval_engine/
  cli.py                    ← CLI entrypoint (6 commands)
  config.py                 ← Pydantic v2 EvalConfig + YAML loader
  runner.py                 ← Async EvalRunner (semaphore, retry, JSONL streaming)
  cost_tracker.py           ← Per-call token + cost accounting, budget enforcement
  sanitizer.py              ← Prompt injection sanitization for judge calls
  statistics.py             ← StatEngine (9 statistical methods + StatReport)
  schemas.py                ← Typed evaluation data structures
  reproducibility.py        ← Seed, git hash, dependency capture per run
  dataset_manifest.py       ← SHA-256 content hashing + lineage tracking
  comparison.py             ← Run diff engine (verity compare)

  metrics/                  ← 74 registered metrics (21 core + 4 rankers + 47 expansion + 2 Tier 3 scaffolds)
  agents/                   ← Proposer → Critic → Judge agents
  orchestration/            ← DebateRound, OversightRunner, Celery tasks
```

---

## Configuration Reference

```yaml
experiment_id: consolidation_eval_run1
architecture: consolidation          # centralized | decentralized | consolidation
model: claude-sonnet-4-6

dataset:
  path: datasets/eval_set.json
  format: json                       # json | jsonl | csv | parquet
  question_col: question
  context_col: contexts
  answer_col: answer
  ground_truth_col: ground_truth
  sample_n: null                     # null = full dataset

metrics:
  - name: ndcg
    enabled: true
    k: 10
  - name: ragas_consolidation_delta
    enabled: true
    pre_consolidation_snapshot: snapshots/pre/
    post_consolidation_snapshot: snapshots/post/
  - name: llamaguard_safety
    enabled: true
    kwargs:
      mode: local
      quantize: true                 # Required for 12GB VRAM GPUs

budget:
  max_usd: 5.00
  warn_at_pct: 0.80
  track_tokens: true

async_cfg:
  max_concurrent_queries: 10
  batch_size: 50
  timeout_seconds: 30.0

retry:
  max_retries: 3
  backoff_base_seconds: 2.0
  backoff_max_seconds: 60.0

statistics:
  tests: [wilcoxon, cohens_d]
  alpha: 0.05
  save_figures: true
  figures_dir: outputs/figures

seed: 42
track: true

output_dir: outputs
notes: "Consolidation delta evaluation — primary run."
```

---

## Output Structure

### Metric evaluation (`verity run`)

```
outputs/{experiment_id}/
  results.jsonl              ← Per-query scores (streamed)
  manifest.json              ← Run summary: scores, duration, success rate
  cost_ledger_{id}.jsonl     ← Per-call token and cost log
  reproducibility.json       ← Seed, git hash, deps (when --track)
  dataset_manifest.json      ← Dataset SHA-256 + lineage (when --track)
```

### Oversight run (`verity oversight-run`)

```
outputs/{experiment_id}/
  oversight_results.jsonl    ← Per-debate verdict, scores, RH flag
  oversight_manifest.json    ← Batch summary: RH rate, verdict distribution
  oversight_stats.json       ← Statistical analysis of safety/accuracy scores
  cost_ledger_{id}.jsonl     ← Per-agent token and cost log
  debates/
    debate_{query_id}.json   ← Full A→B→C trace per query
```

### Comparison (`verity compare`)

```
outputs/
  compare_{run_a}_vs_{run_b}.json
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...       # Required for real API calls
OPENAI_API_KEY=sk-...              # Optional — OpenAI judge models
REDIS_URL=redis://localhost:6379/0 # Celery distributed mode

# Agent model overrides (also settable via CLI)
PROPOSER_MODEL=claude-haiku-4-5
CRITIC_MODEL=claude-sonnet-4-6
JUDGE_MODEL=claude-haiku-4-5
```

---

## Security

Verity has a genuine threat model authored by a CISSP-certified security professional. See [SECURITY.md](SECURITY.md) for:

- Trust boundary diagram
- Prompt injection (OWASP LLM01) mitigation and residual risk
- Excessive agency (OWASP LLM08) controls
- Redis state poisoning risk in distributed mode
- API key exposure controls
- Responsible disclosure policy

---

## Contributing

Verity uses a Contributor License Agreement (CLA). All pull requests require CLA sign-off before merge.

Development setup:

```bash
pip install -e ".[dev]"
python -m pytest eval_engine/tests/ -q    # 245 tests
```

Adding a new metric:

1. Implement `BaseMetric` in `eval_engine/metrics/your_metric.py`
2. Register in `eval_engine/metrics/__init__.py`
3. Add tests in `eval_engine/tests/`
4. Submit PR with CLA sign-off

See [STATUS.md](STATUS.md) for implementation maturity and [ARCHITECTURE.md](ARCHITECTURE.md) for system design before contributing.

---

## License

MIT © Aurelian Security

See [LICENSE](LICENSE) for full terms.
