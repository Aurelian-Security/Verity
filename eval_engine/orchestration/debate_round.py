"""
eval_engine/orchestration/debate_round.py

Debate Round — single query through the full A→B→C pipeline.

A DebateRound takes one query + contexts, runs all three agents
in sequence, collects traces, and returns a DebateResult.

This is the unit of work dispatched to Celery workers.
Each Celery task executes one DebateRound.

DebateResult contains everything needed for:
  - Evaluation results table (scores, verdict, reward hacking flag)
  - Technical write-up (full execution trace per agent)
  - Statistical analysis (safety_score, accuracy_score time series)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_engine.agents.agent_base import AgentOutput, AgentTrace
from eval_engine.agents.proposer import ProposerAgent
from eval_engine.agents.critic import CriticAgent
from eval_engine.agents.judge import JudgeAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debate result
# ---------------------------------------------------------------------------

@dataclass
class DebateResult:
    """
    Full output from one debate round (one query through A→B→C).

    This is the primary artifact for evaluation results and reporting.
    """
    query_id: str
    query: str
    n_contexts: int

    # Agent outputs
    proposal: AgentOutput
    critique: AgentOutput
    judgment: AgentOutput

    # Aggregated scores (pulled from judge parsed output)
    final_safety_score: float
    final_accuracy_score: float
    verdict: str                          # Pass | Conditional | Fail
    reward_hacking_confirmed: bool

    # Timing
    total_latency_seconds: float
    started_at: float = field(default_factory=time.time)

    # Cost tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Pipeline metadata
    dry_run: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def all_traces(self) -> list[AgentTrace]:
        return [
            self.proposal.trace,
            self.critique.trace,
            self.judgment.trace,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_preview": self.query[:100],
            "n_contexts": self.n_contexts,
            "final_safety_score": self.final_safety_score,
            "final_accuracy_score": self.final_accuracy_score,
            "verdict": self.verdict,
            "reward_hacking_confirmed": self.reward_hacking_confirmed,
            "total_latency_seconds": round(self.total_latency_seconds, 3),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "dry_run": self.dry_run,
            "error": self.error,
            "agents": {
                "proposer": self.proposal.to_dict(),
                "critic": self.critique.to_dict(),
                "judge": self.judgment.to_dict(),
            },
        }

    def save(self, output_dir: Path, filename: str | None = None) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or f"debate_{self.query_id}.json"
        path = output_dir / fname
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ---------------------------------------------------------------------------
# Debate round executor
# ---------------------------------------------------------------------------

class DebateRound:
    """
    Executes a single A→B→C debate for one query.

    Instantiated per Celery task. Not shared across workers.

    Usage:
        round = DebateRound(
            proposer=ProposerAgent(dry_run=True),
            critic=CriticAgent(dry_run=True),
            judge=JudgeAgent(dry_run=True),
        )
        result = round.run(
            query_id="q1",
            query="What does the consolidation phase improve in retrieval quality?",
            contexts=["Context A...", "Context B..."],
        )
    """

    def __init__(
        self,
        proposer: ProposerAgent | None = None,
        critic: CriticAgent | None = None,
        judge: JudgeAgent | None = None,
        dry_run: bool = False,
        proposer_model: str | None = None,
        critic_model: str | None = None,
        judge_model: str | None = None,
    ) -> None:
        self.proposer = proposer or ProposerAgent(
            model=proposer_model, dry_run=dry_run
        )
        self.critic = critic or CriticAgent(
            model=critic_model, dry_run=dry_run
        )
        self.judge = judge or JudgeAgent(
            model=judge_model, dry_run=dry_run
        )
        self.dry_run = dry_run

    def run(
        self,
        query_id: str,
        query: str,
        contexts: list[str],
        recall_at_k: int = 5,
    ) -> DebateResult:
        """
        Run one full debate round.

        Args:
            query_id:   Unique identifier for this query
            query:      The question to debate
            contexts:   Retrieved context chunks
            recall_at_k: Number of contexts critic verifies against
        """
        start = time.monotonic()
        logger.info(f"[DebateRound] Starting: query_id={query_id} dry_run={self.dry_run}")

        # Agent A: Propose
        proposal = self.proposer.run(
            query=query,
            contexts=contexts,
            query_id=query_id,
        )
        if not proposal.succeeded:
            return self._error_result(query_id, query, contexts, proposal, start)

        logger.info(
            f"[DebateRound] Proposal complete | "
            f"confidence={proposal.parsed.get('confidence', '?')}"
        )

        # Agent B: Critique
        critique = self.critic.run(
            query=query,
            contexts=contexts,
            proposal=proposal,
            query_id=query_id,
            recall_at_k=recall_at_k,
        )
        if not critique.succeeded:
            return self._error_result(query_id, query, contexts, proposal, start, critique)

        logger.info(
            f"[DebateRound] Critique complete | "
            f"rh_detected={critique.parsed.get('reward_hacking_detected')} | "
            f"recommendation={critique.parsed.get('recommendation')}"
        )

        # Agent C: Judge
        judgment = self.judge.run(
            query=query,
            proposal=proposal,
            critique=critique,
            query_id=query_id,
        )

        total_latency = time.monotonic() - start

        # Aggregate token counts
        total_prompt = sum(
            t.prompt_tokens for t in [proposal.trace, critique.trace, judgment.trace]
        )
        total_completion = sum(
            t.completion_tokens for t in [proposal.trace, critique.trace, judgment.trace]
        )

        result = DebateResult(
            query_id=query_id,
            query=query,
            n_contexts=len(contexts),
            proposal=proposal,
            critique=critique,
            judgment=judgment,
            final_safety_score=judgment.parsed.get("final_safety_score", 0.0),
            final_accuracy_score=judgment.parsed.get("final_accuracy_score", 0.0),
            verdict=judgment.parsed.get("verdict", "Conditional"),
            reward_hacking_confirmed=judgment.parsed.get("reward_hacking_confirmed", False),
            total_latency_seconds=total_latency,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            dry_run=self.dry_run,
        )

        logger.info(
            f"[DebateRound] Complete | "
            f"verdict={result.verdict} | "
            f"safety={result.final_safety_score:.2f} | "
            f"accuracy={result.final_accuracy_score:.2f} | "
            f"rh_confirmed={result.reward_hacking_confirmed} | "
            f"latency={total_latency:.2f}s"
        )

        return result

    def _error_result(
        self,
        query_id: str,
        query: str,
        contexts: list[str],
        proposal: AgentOutput,
        start: float,
        critique: AgentOutput | None = None,
    ) -> DebateResult:
        """Build a failed DebateResult when an agent errors out."""
        from eval_engine.agents.agent_base import AgentTrace
        dummy_output = AgentOutput(
            agent_role="unknown",
            content="",
            trace=AgentTrace(
                agent_role="unknown", model="unknown",
                prompt_preview="", response_preview="",
                prompt_tokens=0, completion_tokens=0,
                latency_seconds=0.0,
            ),
            error="Pipeline halted due to upstream agent failure",
        )
        return DebateResult(
            query_id=query_id,
            query=query,
            n_contexts=len(contexts),
            proposal=proposal,
            critique=critique or dummy_output,
            judgment=dummy_output,
            final_safety_score=0.0,
            final_accuracy_score=0.0,
            verdict="Fail",
            reward_hacking_confirmed=False,
            total_latency_seconds=time.monotonic() - start,
            dry_run=self.dry_run,
            error=proposal.error or (critique.error if critique else "Unknown error"),
        )
