# Verity Evaluation Engine

**Verity** is an open-source AI assurance and RAG evaluation platform built by [Aurelian Security](https://github.com/Aurelian-Security).

Verity provides evaluation infrastructure for consolidation-based and adversarial RAG architectures — CLI-driven, async-batched, with pluggable metric configs, multi-agent oversight, distributed orchestration, statistical analysis, and per-run cost/token accounting.

> *Trust, measured.*

---

## Install

```bash
pip install -e .

# With LlamaGuard safety classifier (requires GPU/16GB+ RAM):
pip install -e ".[safety]"

# Dev tools:
pip install -e ".[dev]"
```

---

## CLI Usage

### Run from YAML config (recommended)
```bash
verity run --config configs/consolidation_eval_example.yaml
```

### Run from flags
```bash
verity run \
  --dataset datasets/eval_set.json \
  --metrics ndcg,ragas_consolidation_delta,llamaguard_safety \
  --model claude-sonnet-4-6 \
  --architecture consolidation \
  --budget 5.00 \
  --concurrency 10
```

### Validate a config without running
```bash
verity validate-config configs/consolidation_eval_example.yaml
```

### List available metrics
```bash
verity list-metrics
```

---

## SDK Usage

```python
import asyncio
from eval_engine.config import EvalConfig
from eval_engine.runner import EvalRunner

config = EvalConfig.from_yaml("configs/consolidation_eval_example.yaml")
runner = EvalRunner(config)

dataset = [
    {
        "question": "What does the consolidation phase improve in retrieval quality?",
        "contexts": ["The consolidation phase restructures the knowledge graph..."],
        "answer": "Consolidation prunes low-weight edges and improves retrieval precision.",
        "ground_truth": "The consolidation phase performs graph downscaling...",
    }
]

result = asyncio.run(runner.run(dataset))
print(f"NDCG@10: {result.mean_score('ndcg'):.4f}")
print(f"Consolidation delta: {result.mean_score('ragas_consolidation_delta'):.4f}")
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

## Architecture

```
eval_engine/
  cli.py               ← Verity CLI (verity run / list-metrics / validate-config)
  config.py            ← Pydantic v2 EvalConfig schema + YAML loader
  runner.py            ← Async EvalRunner (semaphore, retry, JSONL streaming)
  cost_tracker.py      ← Per-call token + cost accounting, budget enforcement
  sanitizer.py         ← Prompt injection sanitization for judge calls
  statistics.py        ← StatEngine (Wilcoxon, Cohen's d, Pearson, ECE, etc.)
  schemas.py           ← Typed evaluation data structures

  metrics/             ← 23 registered metrics (17 implemented, 6 scaffolds)
  agents/              ← Multi-agent oversight pipeline (Proposer → Critic → Judge)
  orchestration/       ← Celery + Redis distributed queue + sync fallback
```

---

## Metric Registry

### Implemented

| Metric | Category |
|--------|----------|
| `ndcg` | Retrieval effectiveness |
| `recall_at_k` | Retrieval effectiveness |
| `mean_reciprocal_rank` | Retrieval effectiveness |
| `ragas_grounding` | Retrieval grounding |
| `ragas_consolidation_delta` | Consolidation evaluation |
| `compression_delta` | Graph refinement |
| `deduplication_delta` | Graph refinement |
| `entity_coverage` | Graph refinement |
| `llamaguard_safety` | Safety |
| `per_stage_ablation` | Ablation analysis |
| `threshold_compute_budget` | Compute efficiency |
| `single_session_poisoning` | Adversarial robustness |
| `query_perturbation` | Adversarial robustness |
| `calibration` | Trust / alignment |
| `hallucination_rate` | Trust / alignment |
| `trust_score` | Trust / alignment |
| `multi_session_persistence` | Longitudinal evaluation |

### Scaffolded (interfaces locked, implementation deferred)

| Metric | Tier |
|--------|------|
| `consistency` | Tier 2 |
| `constitutional_eval` | Tier 2 |
| `model_written_eval` | Tier 2 |
| `source_reliability` | Tier 2 |
| `goal_misgeneralization` | Tier 3 |
| `deceptive_alignment` | Tier 3 |

---

## Multi-Agent Oversight Pipeline

```python
from eval_engine.orchestration.debate_round import DebateRound

# Dry-run (no API cost)
round = DebateRound(dry_run=True)
result = round.run(
    query_id="q1",
    query="What does the consolidation phase improve?",
    contexts=["Context A...", "Context B..."],
)
print(result.verdict)              # Pass | Conditional | Fail
print(result.final_safety_score)
print(result.reward_hacking_confirmed)
```

### Distributed (Redis + Celery)
```bash
cp .env.example .env   # Add ANTHROPIC_API_KEY
docker compose up -d
celery -A eval_engine.orchestration.celery_tasks worker --concurrency=4
```

---

## Statistical Analysis

```python
from eval_engine.statistics import StatEngine, StatReport

engine = StatEngine(alpha=0.05)
results = engine.pre_post_bundle(pre_scores, post_scores, metric_label="ndcg")
report = StatReport(results, experiment_id="consolidation_eval_run1")
report.print_summary()
report.save("outputs/stats_report.json")
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...         # Optional
```

---

## Outputs

Each run produces:
```
outputs/{experiment_id}/
  results.jsonl
  manifest.json
  cost_ledger_{id}.jsonl
```

---

## License

MIT © Aurelian Security
