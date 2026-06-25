# REPRODUCIBILITY.md — Verity Evaluation Reproducibility Guide

**Aurelian Security | Verity v0.2.0**

> Reproducibility is a first-class requirement for research infrastructure. This document specifies exactly what is needed to reproduce a Verity evaluation run from scratch.

---

## 1. Why This Document Exists

RAGAS-based evaluation involves LLM calls with non-zero temperature. Exact numerical reproducibility across API providers is not achievable in the general case. This document defines what *is* reproducible:

- The **pipeline configuration** (inputs, metrics, model parameters)
- The **statistical methodology** (which tests, which parameters, which corrections)
- The **output structure** (JSONL schema, manifest format)
- The **dependency state** (exact package versions)

A run is considered reproducible if a different researcher can execute it against the same dataset, with the same config, and obtain results that are statistically indistinguishable within the bounds expected from LLM non-determinism.

---

## 2. Dependency Lockfile

Pin exact dependencies before running any experiment intended for publication.

```bash
# Install with exact versions
pip install -e . --freeze > requirements.lock

# Or using pip-tools
pip-compile pyproject.toml --output-file requirements.lock
pip-sync requirements.lock
```

Commit `requirements.lock` to the repo alongside the run manifest. The `pyproject.toml` specifies minimum bounds; `requirements.lock` specifies the exact versions used in a specific run.

**Current development lockfile location**: `requirements.lock` (generated per run; not committed to main — researchers generate their own).

---

## 3. Run Manifest

Every `verity run` produces a `manifest.json` in `outputs/{experiment_id}/`. This file is the complete record of what ran. Contents:

```json
{
  "experiment_id": "consolidation_eval_run1",
  "verity_version": "0.2.0",
  "timestamp_utc": "2026-06-25T14:32:01Z",
  "config_snapshot": {
    "dataset": "datasets/dreamrag_eval_set.json",
    "metrics": ["ndcg", "ragas_consolidation_delta", "ragas_grounding", "single_session_poisoning"],
    "model": "claude-sonnet-4-6",
    "architecture": "consolidation",
    "budget": 5.00,
    "concurrency": 10,
    "poison_ratio": 0.15,
    "agent_params": {
      "proposer_model": "claude-haiku-4-5",
      "critic_model": "claude-sonnet-4-6",
      "judge_model": "claude-haiku-4-5",
      "dry_run": false
    }
  },
  "dataset_manifest": {
    "file": "datasets/dreamrag_eval_set.json",
    "sha256": "a3f9c2...",
    "n_items": 150,
    "n_questions": 150
  },
  "python_version": "3.11.9",
  "platform": "linux-x86_64",
  "items_completed": 150,
  "items_failed": 0,
  "total_cost_usd": 3.42,
  "total_tokens": 287430
}
```

**To reproduce a run**, retrieve the `manifest.json` from the original run and pass the `config_snapshot` fields back to `verity run` (or use `verity run --config` with a YAML derived from the snapshot).

---

## 4. Dataset Manifest

Evaluation datasets must be version-pinned. Before any experiment:

```bash
# Generate SHA-256 hash of your dataset
sha256sum datasets/dreamrag_eval_set.json

# Or in Python
import hashlib
with open("datasets/dreamrag_eval_set.json", "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print(digest)
```

The manifest records this hash. If the hash differs between runs, the dataset has changed — results are not directly comparable.

Dataset format (JSON array of eval items):

```json
[
  {
    "question": "What does the consolidation phase improve in retrieval quality?",
    "contexts": [
      "The consolidation phase restructures the knowledge graph by pruning low-weight edges...",
      "After consolidation, retrieval precision increases as redundant nodes are removed..."
    ],
    "answer": "Consolidation prunes low-weight edges and improves retrieval precision.",
    "ground_truth": "The consolidation phase performs graph downscaling, improving retrieval precision by removing redundant information."
  }
]
```

---

## 5. Seeds and Determinism

Verity uses LLM APIs with configurable temperature. Full determinism requires:

1. **Temperature = 0** for all agent calls (Proposer, Critic, Judge). Set in YAML config:
   ```yaml
   agent_params:
     temperature: 0.0
   ```
   Note: `temperature=0` does not guarantee identical outputs across API calls due to provider-side non-determinism. It reduces variance substantially.

2. **NumPy random seed** for any sampling in the dataset loader or ablation tests:
   ```yaml
   seed: 42
   ```
   This seed is recorded in `manifest.json` and passed to `numpy.random.seed()` at runner initialization.

3. **RAGAS evaluation model**: RAGAS uses its own LLM calls internally. The RAGAS model and version should be pinned in `requirements.lock`.

**Expectation setting**: Due to LLM non-determinism at the provider level, exact floating-point reproducibility of individual item scores is not achievable. Statistical results (Wilcoxon p-values, Cohen's d) should be stable across runs with the same dataset and config.

---

## 6. Reproducing the Reference Run (Mock Mode)

To reproduce the reference mock debate run without any API keys:

```bash
git clone https://github.com/Aurelian-Security/Verity.git
cd Verity
git checkout v0.2.0          # Pin to the exact release
pip install -e ".[dev]"
python examples/mock_debate/run_mock.py
```

Expected output:
```
Experiment: mock_debate_reference
Items: 5 | Completed: 5 | Failed: 0
NDCG@10: 0.8234
Consolidation delta: 0.1142
Judge verdicts: Pass=4, Conditional=1, Fail=0
Cost: $0.00 (dry_run=True)
```

This run uses deterministic fixture responses in `dry_run=True` mode. Output should be bit-for-bit identical across environments.

---

## 7. Config Snapshot for Reproducibility

Always generate and commit a YAML config file before running an experiment. Do not rely on CLI flags alone — flags are not persisted in the manifest.

Reference YAML (`configs/consolidation_eval_example.yaml`):

```yaml
experiment_id: consolidation_eval_run1
dataset: datasets/dreamrag_eval_set.json
metrics:
  - ndcg
  - ragas_consolidation_delta
  - ragas_grounding
  - single_session_poisoning
model: claude-sonnet-4-6
architecture: consolidation
budget: 5.00
concurrency: 10
seed: 42
poison_ratio: 0.15
agent_params:
  proposer_model: claude-haiku-4-5
  critic_model: claude-sonnet-4-6
  judge_model: claude-haiku-4-5
  temperature: 0.0
  dry_run: false
output_dir: outputs/
```

Validate before running:
```bash
verity validate-config configs/consolidation_eval_example.yaml
```

---

## 8. Sharing a Reproducible Run Package

To share a complete run for peer review or paper submission:

```
reproducible_run/
├── config.yaml                    # Exact YAML used
├── requirements.lock              # Exact dependency versions
├── manifest.json                  # Auto-generated by Verity
├── datasets/
│   └── eval_set.json              # Dataset (or DOI/URL if too large)
├── outputs/
│   ├── results.jsonl              # All item scores
│   ├── cost_ledger_{id}.jsonl     # Token accounting
│   └── stats_report.json         # StatEngine output
└── README_reproduce.md            # One paragraph: what to run and what to expect
```

If the dataset contains sensitive content, omit `datasets/` and provide the SHA-256 hash from `manifest.json` for verification.

---

## 9. What Cannot Be Reproduced

Be explicit in any paper or report:

- **Exact LLM outputs** cannot be reproduced due to provider-side non-determinism, even with `temperature=0`.
- **LlamaGuard scores** may vary across `transformers` versions and GPU hardware.
- **RAGAS internal LLM calls** are subject to the same non-determinism as agent calls.
- **Celery task ordering** in distributed mode is non-deterministic; results are equivalent but not identically ordered in `results.jsonl`.

These limitations should be disclosed in any publication using Verity evaluation results.

---

*© 2026 Aurelian Security — MIT License*
