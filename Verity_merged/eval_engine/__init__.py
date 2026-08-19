"""Verity Evaluation Engine — AI assurance and RAG evaluation infrastructure."""
from eval_engine.config import EvalConfig
from eval_engine.runner import EvalRunner, RunResult
from eval_engine.cost_tracker import CostTracker, BudgetExceededError
from eval_engine.sanitizer import InputSanitizer
from eval_engine.statistics import StatEngine, StatReport, StatResult
from eval_engine.metrics import registry, MetricsRegistry
from eval_engine.schemas import RetrievalCase, RetrievalResult, GraphSnapshot
from eval_engine.orchestration import OversightRunner, OversightRunResult, DebateRound, DebateResult

__version__ = "0.2.0"
__product__ = "verity"

__all__ = [
    "EvalConfig", "EvalRunner", "RunResult",
    "CostTracker", "BudgetExceededError",
    "InputSanitizer",
    "StatEngine", "StatReport", "StatResult",
    "registry", "MetricsRegistry",
    "RetrievalCase", "RetrievalResult", "GraphSnapshot",
    "OversightRunner", "OversightRunResult",
    "DebateRound", "DebateResult",
]
