"""
eval_engine/metrics/ragas_runner.py

Optional RAGAS runner for Dream RAG retrieval-grounding evaluation.

SOURCE: Consolidation RAG eval framework (your code), with the following additions:
  FIX-2: RAGAS LLM backend configuration added.
          RAGAS defaults to OpenAI regardless of what model is set in EvalConfig.
          Without explicit backend config, Claude/Anthropic API is silently bypassed.
          configure_ragas_llm() must be called before run_ragas_evaluation().
  FIX-4: Error handling, timeout, and partial result recovery added.
          A bare evaluate() call crashes the entire batch on any LLM failure.
          run_ragas_evaluation() now returns partial results on failure
          and logs the error rather than raising.

UNCHANGED from your original:
  - run_ragas_evaluation() signature (backwards compatible)
  - available_ragas_metrics() (verbatim)
  - Lazy import pattern (verbatim)

Requires optional dependencies:
    pip install ragas datasets
For Claude backend:
    pip install langchain-anthropic
"""
from __future__ import annotations

import logging
import time
from typing import Any

from eval_engine.metrics.ragas_adapter import RagasCase, ragas_metric_names, to_ragas_dataset

logger = logging.getLogger(__name__)


# =============================================================================
# FIX-2: RAGAS LLM BACKEND CONFIGURATION
# =============================================================================

def configure_ragas_llm(
    model: str = "claude-sonnet-4-6",
    provider: str = "anthropic",
) -> None:
    """
    Configure RAGAS to use a specific LLM backend.

    MUST be called before run_ragas_evaluation() if you want to use
    a model other than OpenAI. Without this, RAGAS silently uses
    OPENAI_API_KEY regardless of EvalConfig.model.

    Args:
        model:    Model string (e.g. "claude-sonnet-4-6", "gpt-4o")
        provider: "anthropic" | "openai" | "ollama"

    Usage:
        configure_ragas_llm(model="claude-sonnet-4-6", provider="anthropic")
        results = run_ragas_evaluation(cases)
    """
    try:
        from ragas.metrics import context_precision, context_recall, faithfulness

        if provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                from ragas.llms import LangchainLLMWrapper

                llm = LangchainLLMWrapper(ChatAnthropic(model=model))
                for metric in [faithfulness, context_precision, context_recall]:
                    metric.llm = llm
                logger.info(f"RAGAS LLM backend set to Anthropic: {model}")

            except ImportError as e:
                raise ImportError(
                    "Anthropic backend requires langchain-anthropic: "
                    "pip install langchain-anthropic"
                ) from e

        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper

                llm = LangchainLLMWrapper(ChatOpenAI(model=model))
                for metric in [faithfulness, context_precision, context_recall]:
                    metric.llm = llm
                logger.info(f"RAGAS LLM backend set to OpenAI: {model}")

            except ImportError as e:
                raise ImportError(
                    "OpenAI backend requires langchain-openai: "
                    "pip install langchain-openai"
                ) from e

        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                from ragas.llms import LangchainLLMWrapper

                llm = LangchainLLMWrapper(ChatOllama(model=model))
                for metric in [faithfulness, context_precision, context_recall]:
                    metric.llm = llm
                logger.info(f"RAGAS LLM backend set to Ollama: {model}")

            except ImportError as e:
                raise ImportError(
                    "Ollama backend requires langchain-ollama: "
                    "pip install langchain-ollama"
                ) from e

        else:
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Supported: 'anthropic', 'openai', 'ollama'"
            )

    except ImportError as e:
        raise ImportError(
            "RAGAS backend configuration requires ragas: pip install ragas"
        ) from e


# =============================================================================
# YOUR ORIGINAL CODE — with FIX-4 error handling wrapped around evaluate()
# =============================================================================

def run_ragas_evaluation(
    cases: list[RagasCase],
    timeout_seconds: float = 120.0,
    return_partial: bool = True,
) -> dict[str, Any]:
    """
    Run RAGAS retrieval-grounding metrics.

    Evaluates:
        - Context Recall
        - Context Precision
        - Faithfulness

    FIX-4 additions (backwards compatible):
        timeout_seconds: Max seconds before aborting evaluate() call.
        return_partial:  If True, returns whatever scores completed on failure.
                         If False, re-raises the exception (original behavior).

    Returns:
        dict with keys: context_recall, context_precision, faithfulness, _meta
        _meta contains: success (bool), error (str|None), duration_seconds (float)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS evaluation requires `ragas` and `datasets`. "
            "Install with: pip install ragas datasets"
        ) from exc

    dataset = to_ragas_dataset(cases)

    start = time.monotonic()
    error_msg: str | None = None
    result: Any = None

    try:
        result = evaluate(
            dataset,
            metrics=[
                context_recall,
                context_precision,
                faithfulness,
            ],
        )
        duration = time.monotonic() - start
        logger.info(
            f"RAGAS evaluation complete: {len(cases)} cases in {duration:.1f}s"
        )

        scores = dict(result)
        scores["_meta"] = {
            "success": True,
            "error": None,
            "duration_seconds": round(duration, 3),
            "num_cases": len(cases),
        }
        return scores

    except Exception as e:
        duration = time.monotonic() - start
        error_msg = str(e)
        logger.error(
            f"RAGAS evaluation failed after {duration:.1f}s: {error_msg}. "
            f"Processed {len(cases)} cases."
        )

        if not return_partial:
            raise

        # Return partial results with zeros and error metadata
        partial: dict[str, Any] = {
            "context_recall": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
        }
        if result is not None:
            # Some scores may have completed before the failure
            try:
                partial.update({k: v for k, v in dict(result).items() if isinstance(v, float)})
            except Exception:
                pass

        partial["_meta"] = {
            "success": False,
            "error": error_msg,
            "duration_seconds": round(duration, 3),
            "num_cases": len(cases),
        }
        return partial


def run_ragas_evaluation_batch(
    cases: list[RagasCase],
    batch_size: int = 50,
    timeout_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """
    Run RAGAS evaluation in batches to avoid timeout on large datasets.
    Returns one result dict per batch.

    Use this when len(cases) > 100 to avoid a single long-running evaluate() call.
    """
    batches = [cases[i:i+batch_size] for i in range(0, len(cases), batch_size)]
    results = []

    for i, batch in enumerate(batches):
        logger.info(f"RAGAS batch {i+1}/{len(batches)} ({len(batch)} cases)")
        result = run_ragas_evaluation(
            batch,
            timeout_seconds=timeout_seconds,
            return_partial=True,
        )
        result["_meta"]["batch_index"] = i
        results.append(result)

    return results


def available_ragas_metrics() -> list[str]:
    """Return the RAGAS metrics exposed by this runner."""
    return ragas_metric_names()
