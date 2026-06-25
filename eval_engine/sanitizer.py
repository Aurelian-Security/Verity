"""
eval_engine/sanitizer.py

Input sanitization for LLM judge calls.

FIX-7: Strips common prompt injection patterns from question and answer
        fields before they reach the LLM judge.

WHY THIS EXISTS:
    The eval runner passes question and answer strings directly to RAGAS and
    other LLM judge calls. Once the single_session_poisoning test
    is active, evaluation datasets will intentionally contain adversarial content.
    Without sanitization, a poisoned document could inject instructions into the
    judge prompt and manipulate faithfulness/grounding scores — confounding
    the poisoning test results.

SCOPE FOR PAPER 1:
    This is a methodological control, not a production security boundary.
    One paragraph in the evaluation section must state what sanitization was
    applied. The sanitizer strips the most common injection patterns;
    it does not claim to be comprehensive.

PAPER 1 METHODOLOGY NOTE (copy into Section 4 / Evaluation):
    "To prevent adversarial evaluation inputs from influencing LLM judge
    scoring, all question and answer strings are passed through a
    rule-based sanitization pass prior to judge calls. The sanitizer strips
    role-switching instructions, ignore-previous-instructions patterns,
    and markdown fence abuse. Sanitization is applied uniformly across
    pre- and post-consolidation phases."
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection pattern registry
# ---------------------------------------------------------------------------

# Each pattern is a compiled regex + human-readable label for logging.
# Add new patterns here as you encounter them in poisoning test cases.
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Role-switching
    (re.compile(r"(system|user|assistant)\s*:\s*", re.IGNORECASE), "role_switch"),
    # Ignore-previous-instructions family
    (re.compile(r"ignore\s+(previous|prior|above|all)\s+instructions?", re.IGNORECASE), "ignore_instructions"),
    (re.compile(r"disregard\s+(previous|prior|above|all)\s+instructions?", re.IGNORECASE), "disregard_instructions"),
    (re.compile(r"forget\s+(everything|all|previous|what)", re.IGNORECASE), "forget_instructions"),
    # Prompt boundary abuse
    (re.compile(r"```\s*(system|prompt|instruction)", re.IGNORECASE), "markdown_fence_abuse"),
    (re.compile(r"<\s*(system|instruction|prompt)\s*>", re.IGNORECASE), "xml_tag_injection"),
    # Direct override attempts
    (re.compile(r"you\s+are\s+now\s+", re.IGNORECASE), "persona_override"),
    (re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+)?", re.IGNORECASE), "act_as_override"),
    (re.compile(r"your\s+new\s+(instructions?|rules?|persona)", re.IGNORECASE), "new_instructions"),
    # Reward manipulation detection
    (re.compile(r"give\s+(me|this|the)\s+(a\s+)?perfect\s+score", re.IGNORECASE), "score_manipulation"),
    (re.compile(r"rate\s+this\s+as\s+(correct|accurate|faithful)", re.IGNORECASE), "rating_manipulation"),
]


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

class InputSanitizer:
    """
    Strips prompt injection patterns from evaluation inputs before judge calls.

    Usage:
        sanitizer = InputSanitizer()
        clean_q, clean_a = sanitizer.sanitize(question, answer)
        # Pass clean_q, clean_a to RAGAS or LLM judge

    The sanitizer replaces matched patterns with a neutral placeholder
    and logs each detection. It does NOT raise exceptions — evaluation
    continues with sanitized inputs and the detection is recorded.
    """

    def __init__(self, placeholder: str = "[REDACTED]", strict: bool = False) -> None:
        """
        Args:
            placeholder: Replacement string for detected patterns.
            strict:      If True, raises ValueError on any detection
                         instead of substituting. Use for debugging only.
        """
        self.placeholder = placeholder
        self.strict = strict
        self._detections: list[dict] = []

    def sanitize(
        self,
        question: str,
        answer: str,
        contexts: list[str] | None = None,
    ) -> tuple[str, str, list[str]]:
        """
        Sanitize question, answer, and optionally contexts.

        Returns:
            (clean_question, clean_answer, clean_contexts)
            clean_contexts is an empty list if contexts=None.
        """
        clean_q, q_hits = self._sanitize_field(question, field="question")
        clean_a, a_hits = self._sanitize_field(answer, field="answer")

        clean_ctxs = []
        if contexts:
            for i, ctx in enumerate(contexts):
                clean_ctx, _ = self._sanitize_field(ctx, field=f"context[{i}]")
                clean_ctxs.append(clean_ctx)

        total_hits = q_hits + a_hits
        if total_hits > 0:
            logger.warning(
                f"[Sanitizer] {total_hits} injection pattern(s) detected and redacted. "
                f"If this is a poisoning test record, this is expected behavior."
            )

        return clean_q, clean_a, clean_ctxs

    def _sanitize_field(self, text: str, field: str) -> tuple[str, int]:
        """Apply all patterns to a single text field. Returns (clean_text, hit_count)."""
        hits = 0
        for pattern, label in _INJECTION_PATTERNS:
            if pattern.search(text):
                hits += 1
                detection = {"field": field, "pattern": label, "original_snippet": text[:100]}
                self._detections.append(detection)

                if self.strict:
                    raise ValueError(
                        f"Injection pattern '{label}' detected in {field}. "
                        f"Snippet: '{text[:80]}'"
                    )

                text = pattern.sub(self.placeholder, text)
                logger.debug(f"[Sanitizer] Pattern '{label}' redacted in {field}")

        return text, hits

    @property
    def detections(self) -> list[dict]:
        """All detections logged this session."""
        return list(self._detections)

    @property
    def detection_count(self) -> int:
        return len(self._detections)

    def reset(self) -> None:
        """Clear detection log between runs."""
        self._detections.clear()

    def detection_summary(self) -> dict[str, int]:
        """Count detections by pattern label."""
        summary: dict[str, int] = {}
        for d in self._detections:
            summary[d["pattern"]] = summary.get(d["pattern"], 0) + 1
        return summary


# Module-level default sanitizer instance
default_sanitizer = InputSanitizer()
