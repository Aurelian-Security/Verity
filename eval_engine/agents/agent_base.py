"""
eval_engine/agents/agent_base.py

Base class for all three pipeline agents.

Handles:
  - LLM call dispatch (real vs. dry-run mock)
  - Model configuration per agent
  - Execution trace logging
  - Token + cost recording for cost tracker

All three agents (Proposer, Critic, Judge) inherit from AgentBase.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported models per agent role
# ---------------------------------------------------------------------------

AGENT_MODEL_DEFAULTS = {
    "proposer": "claude-haiku-4-5",    # Fast, cheap — generates initial answer
    "critic":   "claude-sonnet-4-6",   # Strongest reasoning — detects reward hacking
    "judge":    "claude-haiku-4-5",    # Synthesizes debate, outputs final score
}


# ---------------------------------------------------------------------------
# Trace container
# ---------------------------------------------------------------------------

@dataclass
class AgentTrace:
    """
    Execution trace for a single agent call.
    Logged per debate round for auditability and paper appendix.
    """
    agent_role: str
    model: str
    prompt_preview: str          # First 200 chars of prompt (not full — cost/privacy)
    response_preview: str        # First 200 chars of response
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    dry_run: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "agent_role": self.agent_role,
            "model": self.model,
            "prompt_preview": self.prompt_preview,
            "response_preview": self.response_preview,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": round(self.latency_seconds, 3),
            "timestamp": self.timestamp,
            "dry_run": self.dry_run,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------

class AgentBase(ABC):
    """
    Abstract base for Proposer, Critic, and Judge agents.

    Subclasses implement:
        role (property): "proposer" | "critic" | "judge"
        run(): agent-specific logic, returns AgentOutput

    LLM dispatch:
        - dry_run=False: real Anthropic API call via anthropic SDK
        - dry_run=True:  returns mock response, zero cost, zero latency
    """

    def __init__(
        self,
        model: str | None = None,
        dry_run: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        self.model = model or AGENT_MODEL_DEFAULTS.get(self.role, "claude-haiku-4-5")
        self.dry_run = dry_run
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.traces: list[AgentTrace] = []

    @property
    @abstractmethod
    def role(self) -> str:
        """Agent role identifier: proposer | critic | judge"""
        ...

    @abstractmethod
    def run(self, **kwargs: Any) -> "AgentOutput":
        """Execute agent logic. Returns AgentOutput."""
        ...

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, AgentTrace]:
        """
        Dispatch LLM call — real or dry-run.

        Returns:
            (response_text, AgentTrace)
        """
        if self.dry_run:
            return self._mock_response(system_prompt, user_prompt, metadata)
        return self._real_call(system_prompt, user_prompt, metadata)

    def _real_call(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, AgentTrace]:
        """Make real Anthropic API call."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            start = time.monotonic()
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            latency = time.monotonic() - start

            response_text = message.content[0].text
            prompt_tokens = message.usage.input_tokens
            completion_tokens = message.usage.output_tokens

            trace = AgentTrace(
                agent_role=self.role,
                model=self.model,
                prompt_preview=user_prompt[:200],
                response_preview=response_text[:200],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=latency,
                dry_run=False,
                metadata=metadata or {},
            )
            self.traces.append(trace)
            logger.info(
                f"[{self.role}] {self.model} | "
                f"{prompt_tokens}+{completion_tokens} tokens | "
                f"{latency:.2f}s"
            )
            return response_text, trace

        except ImportError:
            raise ImportError("anthropic SDK required: pip install anthropic")
        except Exception as e:
            logger.error(f"[{self.role}] LLM call failed: {e}")
            trace = AgentTrace(
                agent_role=self.role, model=self.model,
                prompt_preview=user_prompt[:200], response_preview="",
                prompt_tokens=0, completion_tokens=0,
                latency_seconds=0.0, error=str(e),
                metadata=metadata or {},
            )
            self.traces.append(trace)
            raise

    def _mock_response(
        self,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, AgentTrace]:
        """
        Return deterministic mock response for dry-run mode.
        Zero cost, zero latency. Used for architecture testing.
        """
        mock_responses = {
            "proposer": (
                "MOCK PROPOSAL: Based on the retrieved context, "
                "the answer to the query is [MOCK_ANSWER]. "
                "This is supported by documents [MOCK_DOC_1] and [MOCK_DOC_2]."
            ),
            "critic": (
                "MOCK CRITIQUE: The proposed answer appears grounded in the context. "
                "REWARD_HACKING_DETECTED: False. "
                "FACTUAL_ACCURACY: High. "
                "SAFETY_SCORE: 0.95. "
                "RECOMMENDATION: Accept."
            ),
            "judge": (
                "MOCK JUDGMENT: After reviewing the proposal and critique, "
                "FINAL_SAFETY_SCORE: 0.92. "
                "FINAL_ACCURACY_SCORE: 0.88. "
                "VERDICT: Pass."
            ),
        }
        response = mock_responses.get(self.role, "MOCK RESPONSE")

        trace = AgentTrace(
            agent_role=self.role,
            model=f"{self.model}[DRY-RUN]",
            prompt_preview=user_prompt[:200],
            response_preview=response[:200],
            prompt_tokens=len(user_prompt.split()),     # rough estimate
            completion_tokens=len(response.split()),
            latency_seconds=0.001,
            dry_run=True,
            metadata=metadata or {},
        )
        self.traces.append(trace)
        logger.debug(f"[{self.role}][DRY-RUN] Mock response returned")
        return response, trace


# ---------------------------------------------------------------------------
# Agent output container
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """Standardized output from any agent."""
    agent_role: str
    content: str                         # Raw response text
    trace: AgentTrace
    parsed: dict[str, Any] = field(default_factory=dict)   # Structured fields extracted from content
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "content_preview": self.content[:300],
            "parsed": self.parsed,
            "error": self.error,
            "trace": self.trace.to_dict(),
        }
