"""
eval_engine/cost_tracker.py

Per-run LLM judge cost and token accounting.

Intercepts every LLM API call made by the eval runner, logs:
  - prompt tokens
  - completion tokens
  - model ID
  - timestamp
  - call latency

Computes running dollar cost using a static pricing table.
Halts execution if projected cost exceeds EvalConfig.budget.max_usd.
Emits cost summary at run end.

JSONL ledger written to: {output_dir}/cost_ledger_{experiment_id}.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens) — update as models change
# ---------------------------------------------------------------------------

PRICING_TABLE: dict[str, dict[str, float]] = {
    # Anthropic — source: anthropic.com/pricing (verify before use)
    "claude-sonnet-4-6":    {"input": 3.00,   "output": 15.00},
    "claude-haiku-4-5":     {"input": 0.80,   "output": 4.00},
    # OpenAI — source: platform.openai.com/pricing (verify before use)
    "gpt-4o":               {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":          {"input": 0.15,   "output": 0.60},
    # Ollama local — zero cost
    "ollama/mistral":       {"input": 0.0,    "output": 0.0},
    "ollama/llama3":        {"input": 0.0,    "output": 0.0},
}

FALLBACK_PRICING: dict[str, float] = {"input": 5.00, "output": 15.00}


def get_price_per_token(model: str) -> tuple[float, float]:
    """
    Return (input_price, output_price) in USD per token.
    Falls back to GPT-4o pricing if model not in table.
    """
    pricing = PRICING_TABLE.get(model, FALLBACK_PRICING)
    return pricing["input"] / 1_000_000, pricing["output"] / 1_000_000


# ---------------------------------------------------------------------------
# Ledger entry
# ---------------------------------------------------------------------------


@dataclass
class CostEntry:
    timestamp: float
    model: str
    metric_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    cost_usd: float
    call_id: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """
    Tracks LLM API call costs across an eval run.

    Usage:
        tracker = CostTracker(budget_usd=5.00, warn_at_pct=0.80)

        with tracker.track("ragas_consolidation_delta", "claude-sonnet-4-6") as t:
            response = call_llm(...)
            t.record(prompt_tokens=response.usage.input_tokens,
                     completion_tokens=response.usage.output_tokens)

        tracker.check_budget()  # raises BudgetExceededError if over limit
        tracker.save_ledger(output_dir, experiment_id)
        tracker.print_summary()
    """

    def __init__(
        self,
        budget_usd: float = 5.00,
        warn_at_pct: float = 0.80,
    ) -> None:
        self.budget_usd = budget_usd
        self.warn_at_pct = warn_at_pct
        self._entries: list[CostEntry] = []
        self._total_cost: float = 0.0
        self._warned: bool = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        metric_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float = 0.0,
        call_id: str = "",
        error: str | None = None,
    ) -> CostEntry:
        """Record a single LLM API call."""
        input_price, output_price = get_price_per_token(model)
        cost = (prompt_tokens * input_price) + (completion_tokens * output_price)

        entry = CostEntry(
            timestamp=time.time(),
            model=model,
            metric_name=metric_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency_seconds,
            cost_usd=cost,
            call_id=call_id,
            error=error,
        )
        self._entries.append(entry)
        self._total_cost += cost

        logger.debug(
            f"[CostTracker] {metric_name} | {model} | "
            f"{prompt_tokens}+{completion_tokens} tokens | "
            f"${cost:.6f} | running total: ${self._total_cost:.4f}"
        )

        self._check_warn()
        return entry

    def _check_warn(self) -> None:
        if not self._warned and self._total_cost >= self.budget_usd * self.warn_at_pct:
            pct = (self._total_cost / self.budget_usd) * 100
            logger.warning(
                f"[CostTracker] Budget warning: ${self._total_cost:.4f} spent "
                f"({pct:.1f}% of ${self.budget_usd:.2f} limit)"
            )
            self._warned = True

    def check_budget(self) -> None:
        """Raise BudgetExceededError if total cost exceeds budget."""
        if self._total_cost > self.budget_usd:
            raise BudgetExceededError(
                f"Run halted: ${self._total_cost:.4f} exceeds budget of ${self.budget_usd:.2f}"
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def total_prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self._entries)

    @property
    def total_completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self._entries)

    @property
    def total_calls(self) -> int:
        return len(self._entries)

    def summary(self) -> dict[str, Any]:
        by_metric: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            if e.metric_name not in by_metric:
                by_metric[e.metric_name] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            by_metric[e.metric_name]["calls"] += 1
            by_metric[e.metric_name]["cost_usd"] += e.cost_usd
            by_metric[e.metric_name]["tokens"] += e.prompt_tokens + e.completion_tokens

        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "budget_usd": self.budget_usd,
            "budget_remaining_usd": round(self.budget_usd - self._total_cost, 6),
            "budget_utilization_pct": round((self._total_cost / self.budget_usd) * 100, 2),
            "by_metric": by_metric,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 55)
        print("  COST SUMMARY")
        print("=" * 55)
        print(f"  Total calls:       {s['total_calls']}")
        print(f"  Prompt tokens:     {s['total_prompt_tokens']:,}")
        print(f"  Completion tokens: {s['total_completion_tokens']:,}")
        print(f"  Total cost:        ${s['total_cost_usd']:.4f}")
        print(f"  Budget:            ${s['budget_usd']:.2f}")
        print(f"  Budget used:       {s['budget_utilization_pct']:.1f}%")
        print("-" * 55)
        for metric, data in s["by_metric"].items():
            print(f"  {metric:<35} ${data['cost_usd']:.4f} ({data['calls']} calls)")
        print("=" * 55 + "\n")

    def save_ledger(self, output_dir: Path, experiment_id: str) -> Path:
        """Write full per-call ledger to JSONL."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = output_dir / f"cost_ledger_{experiment_id}.jsonl"
        with open(ledger_path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        logger.info(f"Cost ledger saved: {ledger_path}")
        return ledger_path


class BudgetExceededError(RuntimeError):
    """Raised when total LLM spend exceeds configured budget ceiling."""
    pass
