# Verity

**Verity is an asynchronous, multi-agent alignment evaluation and threat modeling infrastructure suite for RAG architectures** — CLI-driven, statistically rigorous, and built for AI assurance research.

[![CI](https://github.com/Aurelian-Security/Verity/actions/workflows/ci.yml/badge.svg)](https://github.com/Aurelian-Security/Verity/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green)](https://docs.pydantic.dev/latest/)
[![Status: Pre-1.0 Research Infrastructure](https://img.shields.io/badge/status-pre--1.0%20research-orange)](STATUS.md)

> *Trust, measured.*

---

> **Maturity disclaimer:** Verity is pre-1.0 open-source research infrastructure. It is not a finished enterprise product. See [STATUS.md](STATUS.md) for what is implemented, experimental, and planned.

---

## Research Use Cases

- **RAG grounding evaluation** — RAGAS faithfulness, answer relevance, and context precision at scale
- **Memory consolidation evaluation** — Pre/post consolidation NDCG@10 delta (DreamRAG-compatible)
- **Poisoning robustness testing** — Configurable `poison_ratio` adversarial injection evaluation
- **Query perturbation robustness** — Semantic and syntactic perturbation stress tests
- **Multi-agent oversight experiments** — Proposer → Critic → Judge debate pipeline with reward hacking detection
- **Ablation analysis** — Per-stage metric ablation across consolidation phases
- **Compute budget sweeps** — Threshold and concurrency sweep across architecture variants

---

## Quickstart (60 seconds, no API key required)

```bash
git clone https://github.com/Aurelian-Security/Verity.git
cd Verity
pip install -e ".[dev]"

# Run mock debate loop — zero API cost, deterministic output
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

### With API keys (live run)

```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

verity run --config configs/consolidation_eval_example.yaml
```

### Install options

```bash
# Base install
pip install -e .

# With LlamaGuard safety classifier (requires GPU / 16GB+ VRAM)
pip install -e ".[safety]"

# With specific LLM backend
pip install -e ".[openai-backend]"    # OpenAI
pip install -e ".[ollama-backend]"    # Local Ollama

# Dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

---

## Declarative Config

All evaluation parameters are configured via YAML. No Python required for standard runs.

```yaml
# configs/consolidation_eval_example.yaml

experiment_id: consolidation_eval_run1
dataset: datasets/dreamrag_eval_set.json

metrics:
  - ndcg
  - ragas_consolidation_delta
  - ragas_grounding
  - single_session_poisoning

model: claude-sonnet-4-6
architecture: consolidation

# Budget ceiling — aborts cleanly if exceeded
budget: 5.00
concurrency: 10
seed: 42

# Adversarial injection ratio for poisoning tests
poison_ratio: 0.15

agent_params:
  proposer_model: claude-haiku-4-5
  critic_model: claude-sonnet-4-6
  judge_model: claude-haiku-4-5
  temperature: 0.0
  dry_run: false

output_dir: outputs/
```

```bash
# Validate without running
verity validate-config configs/consolidation_eval_example.yaml

# Run
verity run --config configs/consolidation_eval_example.yaml

# List all registered metrics
verity list-metrics
```

---

## Capability Matrix

### Phase 1 — Core Evaluation Infrastructure

| Capability | Status |
|---|---|
| CLI entrypoint (`verity run`, `validate-config`, `list-metrics`) | ✅ |
| Pydantic v2 EvalConfig (YAML + flag override) | ✅ |
| Async EvalRunner (semaphore + exponential-backoff retry) | ✅ |
| JSONL streaming output (per-item, not batched) | ✅ |
| Cost tracking + budget ceiling (`BudgetExceededError`) | ✅ |
| Run manifest (config snapshot + dataset SHA-256 + timestamp) | ✅ |
| Mock / dry-run mode (zero API cost) | ✅ |
| Metric plugin registry (`registry.register()`) | ✅ |
| 17 implemented metrics (retrieval, grounding, safety, adversarial, alignment) | ✅ |
| `ndcg`, `recall_at_k`, `mean_reciprocal_rank` | ✅ |
| `ragas_grounding`, `ragas_consolidation_delta` | ✅ |
| `single_session_poisoning`, `query_perturbation` | ✅ |
| `llamaguard_safety` (optional, GPU-only) | ✅ |
| `calibration`, `hallucination_rate`, `trust_score` | ✅ |

### Phase 2 — Multi-Agent Oversight & Statistical Analysis

| Capability | Status |
|---|---|
| Proposer → Critic → Judge debate pipeline | ✅ |
| Reward hacking detection (`reward_hacking_confirmed`) | ✅ |
| Prompt injection sanitizer (pattern-based) | ✅ |
| Wilcoxon rank-sum (pre/post consolidation delta) | ✅ |
| Cohen's d (with small-N and zero-variance guards) | ✅ |
| Pearson correlation + ECE | ✅ |
| StatReport (JSON + Rich summary table) | ✅ |
| Celery + Redis distributed mode | ✅ |
| Sync fallback (no Redis required) | ✅ |
| Flower monitor UI | ✅ |

### Phase 3 — Advanced Alignment Metrics (Partial / Planned)

| Capability | Status |
|---|---|
| `consistency`, `constitutional_eval`, `model_written_eval` | 🔲 Scaffolded |
| `source_reliability` | 🔲 Scaffolded |
| `goal_misgeneralization`, `deceptive_alignment` | 🔲 Scaffolded |
| Bootstrap confidence intervals | 🔲 Planned v0.3 |
| Multiple comparison correction | 🔲 Planned v0.3 |
| Context redaction (`--redact-contexts`) | 🔲 Planned v0.3 |

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
print(result.verdict)               # Pass | Conditional | Fail
print(result.final_safety_score)
print(result.reward_hacking_confirmed)
```

### Distributed mode (Redis + Celery)

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

## Provider Support

| Provider | Extra | Env Var |
|---|---|---|
| Anthropic (default) | base install | `ANTHROPIC_API_KEY` |
| OpenAI | `pip install -e ".[openai-backend]"` | `OPENAI_API_KEY` |
| Ollama (local, air-gapped) | `pip install -e ".[ollama-backend]"` | None |

Proposer, Critic, and Judge models can be set independently in `agent_params` for cost-optimized heterogeneous configurations.

---

## Outputs

Each run produces:

```
outputs/{experiment_id}/
  results.jsonl           ← Per-item scores, agent verdicts, token counts
  manifest.json           ← Config snapshot, dataset hash, run metadata
  cost_ledger_{id}.jsonl  ← Per-call token + cost accounting
```

---

## Known Limitations

- Not validated at large scale (>50 concurrent workers)
- DreamRAG empirical results pending publication
- No GUI; CLI + SDK only
- `llamaguard_safety` requires 16GB+ VRAM (excluded from base install and CI)
- Prompt injection sanitizer is pattern-based, not adversarially robust
- No formal security audit

See [STATUS.md](STATUS.md) for the complete maturity matrix.

---

## Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, component rationale |
| [SECURITY.md](SECURITY.md) | Threat model, OWASP LLM08, controls, disclosure policy |
| [STATUS.md](STATUS.md) | Implementation maturity matrix |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Seeds, lockfiles, manifests, how to reproduce a run |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add metrics, agents, providers |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## License

MIT © [Aurelian Security](https://github.com/Aurelian-Security)
