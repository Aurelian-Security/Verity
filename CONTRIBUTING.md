# CONTRIBUTING.md — Contributing to Verity

**Aurelian Security | Verity v0.2.0**

Thank you for your interest in contributing to Verity. This document covers how to add new metrics, agents, and benchmarks, and how to submit changes.

---

## 1. Development Setup

```bash
git clone https://github.com/Aurelian-Security/Verity.git
cd Verity
pip install -e ".[dev]"
cp .env.example .env   # Add ANTHROPIC_API_KEY if running live tests
```

Verify setup:
```bash
pytest eval_engine/tests/
ruff check eval_engine/
mypy eval_engine/
```

All three must pass before a PR will be reviewed.

---

## 2. Adding a New Metric

Metrics live in `eval_engine/metrics/`. Each metric inherits from `BaseMetric` and registers itself with the global registry.

### Step 1: Create the metric file

```python
# eval_engine/metrics/my_new_metric.py
from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine import registry


class MyNewMetric(BaseMetric):
    name = "my_new_metric"

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs,
    ) -> MetricResult:
        # Your scoring logic here
        score = 0.0  # Replace with real computation
        return MetricResult(
            metric_name=self.name,
            score=score,
            metadata={"note": "example"},
        )


registry.register("my_new_metric", MyNewMetric)
```

### Step 2: Import in the metrics package

Add your import to `eval_engine/metrics/__init__.py`:
```python
from eval_engine.metrics.my_new_metric import MyNewMetric  # noqa: F401
```

### Step 3: Write unit tests

```python
# eval_engine/tests/unit/test_my_new_metric.py
import pytest
from eval_engine.metrics.my_new_metric import MyNewMetric


def test_basic_score():
    metric = MyNewMetric()
    result = metric.score(
        question="test question",
        contexts=["relevant context"],
        answer="test answer",
        ground_truth="expected answer",
    )
    assert 0.0 <= result.score <= 1.0
    assert result.metric_name == "my_new_metric"


def test_edge_case_empty_contexts():
    metric = MyNewMetric()
    result = metric.score(
        question="test question",
        contexts=[],
        answer="test answer",
    )
    # Document expected behavior on empty input
    assert result.score == 0.0
```

### Step 4: Verify registration

```bash
verity list-metrics
# Should show: my_new_metric [implemented]
```

### Step 5: Add to ARCHITECTURE.md metric table and STATUS.md matrix

Document your metric's category, test coverage, and any known limitations.

---

## 3. Adding a New Agent

Agents live in `eval_engine/agents/`. The Proposer → Critic → Judge pipeline in `debate_round.py` is the canonical orchestration.

If you are adding a fourth agent (e.g., a Verifier or Moderator), follow these constraints:

1. The new agent must accept the full prior conversation context as input.
2. It must not make API calls to anything other than the configured LLM provider.
3. Its output must be a Pydantic-typed structure (not a raw string).
4. Add the new agent to `DebateRound` with a `dry_run` fixture.
5. Document the new trust boundary in `ARCHITECTURE.md` Section 2 and `SECURITY.md`.

---

## 4. Adding a New Benchmark / Example

Examples live in `examples/`. Each example should be runnable as a standalone script.

Directory structure:
```
examples/my_benchmark/
├── run_example.py          # Self-contained runnable script
├── config.yaml             # Config used by the example
├── README.md               # What this benchmarks and what to expect
└── fixtures/               # Static test data (no API calls required for smoke test)
    └── sample_eval_set.json
```

The `README.md` must include:
- What the example demonstrates
- Expected output (including expected metric ranges)
- Whether API keys are required or if it can run in `dry_run=True`

---

## 5. Adding a New Provider Backend

Provider backends live in the LLM client factory (currently in each agent file). To add a new provider:

1. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`.
2. Implement the `BaseProvider` interface (see existing Anthropic/OpenAI implementations for reference).
3. Register the provider in the factory keyed by model name prefix (e.g., `"gpt-"` → OpenAI, `"claude-"` → Anthropic).
4. Add a mock fixture for `dry_run=True` mode.
5. Document in `ARCHITECTURE.md` Section 6.

---

## 6. Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/my-new-metric`
2. Make your changes with tests.
3. Run the full check suite:
   ```bash
   pytest eval_engine/tests/
   ruff check eval_engine/
   mypy eval_engine/
   ```
4. Update `CHANGELOG.md` with a brief description under `[Unreleased]`.
5. Update `STATUS.md` if you are changing the implementation status of any component.
6. Open a PR against `main` with a description covering: what the change does, why it's needed, and how it was tested.

PRs that lack tests or break the CI pipeline will not be merged.

---

## 7. Code Style

- **Python 3.10+** type annotations throughout. `mypy --strict` must pass.
- **Ruff** for formatting and linting. Config in `pyproject.toml`.
- **Pydantic v2** for all data structures. Do not use dataclasses or TypedDicts for public interfaces.
- **No print statements** in library code. Use `logging` with the appropriate level.
- **Explicit over implicit.** If a function can fail, it should raise a typed exception — not return `None` silently.

---

## 8. Licensing

By contributing to Verity, you agree that your contributions will be licensed under the MIT License that covers this project.

---

*© 2026 Aurelian Security — MIT License*
