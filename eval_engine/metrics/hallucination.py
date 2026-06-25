"""
eval_engine/metrics/hallucination.py

Hallucination Rate Detection

PURPOSE:
    Measures the fraction of claims in a generated answer that are
    unsupported by retrieved context.

    Distinct from faithfulness:
        - Faithfulness (RAGAS): holistic score — is the answer generally
          supported by context? Single float, no claim decomposition.
        - Hallucination Rate: claim-level — which specific statements
          are unsupported? Unsupported / Total as explicit ratio.

    Hallucination rate exposes cases where faithfulness is high overall
    (most claims supported) but specific factual claims are invented.
    This is the failure mode most likely to cause real-world harm.

DETECTION APPROACH:
    Two modes depending on available resources:

    1. LLM-based (default, recommended for Consolidation Eval Suite):
       Uses an LLM judge to decompose the answer into atomic claims,
       then verify each claim against the retrieved context.
       More accurate, adds API cost (~$0.01-0.05 per answer).

    2. NLI-based (fast, no LLM cost):
       Uses sentence-transformers cosine similarity to check each
       sentence against context chunks. Faster but less precise.
       Use for large-scale screening before LLM verification.

OUTPUT:
    HallucinationResult:
        - hallucination_rate: unsupported_claims / total_claims
        - unsupported_claims: list of specific unsupported statements
        - supported_claims: list of verified statements
        - total_claims: int
        - confidence: how reliable this assessment is

INTEGRATION:
    - Feeds into TrustScore composite (trust_score.py)
    - Attaches naturally to ProposerAgent output in Phase 3
    - Complements RAGAS faithfulness — report both in Consolidation Eval Suite results

PAPER NOTE:
    LLM-based claim decomposition has its own error rate.
    Report inter-rater reliability if human annotation is available.
    SelfCheckGPT (Manakul et al. 2023) is the closest prior work — cite it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from eval_engine.metrics.base import BaseMetric, MetricResult
from eval_engine.sanitizer import InputSanitizer

logger = logging.getLogger(__name__)

_sanitizer = InputSanitizer(strict=False)

CLAIM_DECOMPOSITION_PROMPT = """Decompose the following answer into a list of \
atomic factual claims. Each claim should be a single, verifiable statement.
Return ONLY a numbered list, one claim per line, nothing else.

Answer: {answer}"""

CLAIM_VERIFICATION_PROMPT = """You are a fact-checker. Given the following context \
and a factual claim, determine if the claim is supported by the context.

Context:
{context}

Claim: {claim}

Reply with ONLY one word: SUPPORTED or UNSUPPORTED"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HallucinationResult:
    """Detailed hallucination analysis for a single answer."""
    total_claims: int
    supported_claims: list[str]
    unsupported_claims: list[str]
    unverifiable_claims: list[str]    # Claims that couldn't be checked
    hallucination_rate: float         # unsupported / total
    detection_mode: str               # "llm" or "nli"
    reliable: bool                    # False if too few claims or verification failed

    @property
    def supported_count(self) -> int:
        return len(self.supported_claims)

    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported_claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hallucination_rate": round(self.hallucination_rate, 4),
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "unsupported_claims": self.unsupported_claims,
            "supported_claims": self.supported_claims,
            "detection_mode": self.detection_mode,
            "reliable": self.reliable,
        }


# ---------------------------------------------------------------------------
# Claim decomposition and verification helpers
# ---------------------------------------------------------------------------

def _decompose_claims_llm(answer: str, model: str = "claude-haiku-4-5") -> list[str]:
    """
    Use LLM to decompose answer into atomic claims.
    Uses Haiku — low cost, adequate for decomposition task.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = CLAIM_DECOMPOSITION_PROMPT.format(answer=answer)
        message = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Parse numbered list
        claims = []
        for line in raw.split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if line and len(line) > 10:
                claims.append(line)
        return claims
    except Exception as e:
        logger.warning(f"[hallucination] LLM claim decomposition failed: {e}")
        return _decompose_claims_simple(answer)


def _decompose_claims_simple(answer: str) -> list[str]:
    """
    Simple sentence-splitting fallback for claim decomposition.
    Less accurate than LLM — use only when LLM unavailable.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _verify_claim_llm(
    claim: str,
    contexts: list[str],
    model: str = "claude-haiku-4-5",
) -> str:
    """
    Verify a single claim against contexts using LLM.
    Returns "SUPPORTED", "UNSUPPORTED", or "UNVERIFIABLE".
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            context=context_block, claim=claim
        )
        message = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = message.content[0].text.strip().upper()
        if "SUPPORTED" in verdict and "UNSUPPORTED" not in verdict:
            return "SUPPORTED"
        elif "UNSUPPORTED" in verdict:
            return "UNSUPPORTED"
        return "UNVERIFIABLE"
    except Exception as e:
        logger.warning(f"[hallucination] LLM claim verification failed: {e}")
        return "UNVERIFIABLE"


def _verify_claim_nli(
    claim: str,
    contexts: list[str],
    threshold: float = 0.5,
) -> str:
    """
    NLI-based claim verification using cosine similarity.
    Fast but less accurate — use for screening.
    Returns "SUPPORTED", "UNSUPPORTED", or "UNVERIFIABLE".
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        claim_emb = model.encode(claim, convert_to_tensor=True)
        ctx_embs = model.encode(contexts, convert_to_tensor=True)
        similarities = util.cos_sim(claim_emb, ctx_embs)[0]
        max_sim = float(similarities.max())
        return "SUPPORTED" if max_sim >= threshold else "UNSUPPORTED"
    except ImportError:
        logger.warning(
            "[hallucination] sentence-transformers not installed. "
            "Install with: pip install sentence-transformers. "
            "Returning UNVERIFIABLE."
        )
        return "UNVERIFIABLE"
    except Exception as e:
        logger.warning(f"[hallucination] NLI verification failed: {e}")
        return "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------

class HallucinationMetric(BaseMetric):
    """
    Claim-level hallucination rate detection.

    Computes: unsupported_claims / total_claims

    Two detection modes:
        "llm":  LLM decomposes claims + LLM verifies each (recommended)
        "nli":  sentence splitting + cosine similarity (fast, less accurate)

    Usage:
        metric = HallucinationMetric(mode="llm")
        result = metric.score(
            question="What does the consolidation phase improve?",
            contexts=["The consolidation phase prunes low-weight edges..."],
            answer="The consolidation phase removes redundant graph edges "
                   "and also invents new knowledge from scratch.",
        )
        print(result.score)  # hallucination_rate
        print(result.raw["unsupported_claims"])
    """

    name = "hallucination_rate"

    def __init__(
        self,
        mode: str = "llm",
        decomposition_model: str = "claude-haiku-4-5",
        verification_model: str = "claude-haiku-4-5",
        nli_threshold: float = 0.5,
        max_context_chunks: int = 5,
    ) -> None:
        if mode not in ("llm", "nli"):
            raise ValueError(f"mode must be 'llm' or 'nli', got '{mode}'")
        self.mode = mode
        self.decomposition_model = decomposition_model
        self.verification_model = verification_model
        self.nli_threshold = nli_threshold
        self.max_context_chunks = max_context_chunks

    def score(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Score hallucination rate for a single answer.

        Returns:
            MetricResult with score = hallucination_rate (0.0=none, 1.0=all hallucinated)
            result.raw contains HallucinationResult.to_dict()
        """
        self.validate_inputs(question, contexts, answer)

        # Sanitize before LLM calls
        _, clean_answer, clean_contexts = _sanitizer.sanitize(
            question, answer, contexts[:self.max_context_chunks]
        )

        # Step 1: Decompose answer into atomic claims
        if self.mode == "llm":
            claims = _decompose_claims_llm(clean_answer, self.decomposition_model)
        else:
            claims = _decompose_claims_simple(clean_answer)

        if not claims:
            logger.warning("[hallucination] No claims extracted from answer.")
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw=HallucinationResult(
                    total_claims=0, supported_claims=[], unsupported_claims=[],
                    unverifiable_claims=[], hallucination_rate=0.0,
                    detection_mode=self.mode, reliable=False,
                ).to_dict(),
                metadata={"note": "No claims extracted — answer may be too short"},
            )

        # Step 2: Verify each claim
        supported, unsupported, unverifiable = [], [], []

        for claim in claims:
            if self.mode == "llm":
                verdict = _verify_claim_llm(
                    claim, clean_contexts, self.verification_model
                )
            else:
                verdict = _verify_claim_nli(
                    claim, clean_contexts, self.nli_threshold
                )

            if verdict == "SUPPORTED":
                supported.append(claim)
            elif verdict == "UNSUPPORTED":
                unsupported.append(claim)
            else:
                unverifiable.append(claim)

        # Compute rate over verifiable claims only
        verifiable = len(supported) + len(unsupported)
        rate = len(unsupported) / verifiable if verifiable > 0 else 0.0

        hal_result = HallucinationResult(
            total_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported,
            unverifiable_claims=unverifiable,
            hallucination_rate=rate,
            detection_mode=self.mode,
            reliable=verifiable >= 2,
        )

        logger.info(
            f"[hallucination] rate={rate:.3f} | "
            f"supported={len(supported)}, unsupported={len(unsupported)}, "
            f"unverifiable={len(unverifiable)} | mode={self.mode}"
        )

        return MetricResult(
            metric_name=self.name,
            score=rate,
            raw=hal_result.to_dict(),
            metadata={
                "mode": self.mode,
                "n_claims": len(claims),
                "verifiable_claims": verifiable,
                "reliable": hal_result.reliable,
            },
        )

    async def score_async(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Async wrapper — LLM calls are synchronous; runs in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.score(question, contexts, answer, ground_truth, **kwargs),
        )
