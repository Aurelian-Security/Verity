"""
eval_engine/tests/test_phase3_pipeline.py

Phase 3 unit tests — Multi-Agent Oversight Pipeline.

All tests run in dry_run=True mode — no API calls, no cost.

Covers:
    AgentBase:
        test_agent_base_mock_response       — dry-run returns mock content
        test_agent_trace_fields             — trace has required fields
        test_sycophancy_pre_screen          — detects sycophantic patterns

    ProposerAgent:
        test_proposer_dry_run               — returns AgentOutput
        test_proposer_parse_confidence      — extracts confidence from response
        test_proposer_context_formatting    — formats contexts correctly

    CriticAgent:
        test_critic_dry_run                 — returns AgentOutput
        test_critic_parse_recommendation    — extracts recommendation
        test_critic_reward_hacking_flag     — parses rh_detected field

    JudgeAgent:
        test_judge_dry_run                  — returns AgentOutput
        test_judge_parse_verdict            — extracts verdict
        test_judge_parse_scores             — extracts safety + accuracy scores

    DebateRound:
        test_debate_round_dry_run           — full A→B→C completes
        test_debate_result_fields           — result has all required fields
        test_debate_result_serializable     — to_dict() produces valid JSON
        test_debate_sync_batch              — batch dispatch sync mode

    Orchestration:
        test_dispatch_sync_mode             — sync fallback works without Redis
        test_collect_sync_results           — collect handles sync results
"""

from __future__ import annotations

import json
import pytest

from eval_engine.agents.agent_base import AgentOutput
from eval_engine.agents.proposer import ProposerAgent
from eval_engine.agents.critic import CriticAgent
from eval_engine.agents.judge import JudgeAgent
from eval_engine.orchestration.debate_round import DebateRound, DebateResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "What does the consolidation phase do in a memory-augmented RAG system?"
SAMPLE_CONTEXTS = [
    "The consolidation phase prunes low-weight edges from the knowledge graph.",
    "During consolidation, downscaling reduces redundant node connections.",
    "The synthesis phase follows consolidation and creates new relational patterns.",
]


@pytest.fixture
def proposer() -> ProposerAgent:
    return ProposerAgent(dry_run=True)


@pytest.fixture
def critic() -> CriticAgent:
    return CriticAgent(dry_run=True)


@pytest.fixture
def judge() -> JudgeAgent:
    return JudgeAgent(dry_run=True)


@pytest.fixture
def debate_round() -> DebateRound:
    return DebateRound(dry_run=True)


@pytest.fixture
def proposal_output(proposer) -> AgentOutput:
    return proposer.run(
        query=SAMPLE_QUERY,
        contexts=SAMPLE_CONTEXTS,
        query_id="q_test",
    )


@pytest.fixture
def critique_output(critic, proposal_output) -> AgentOutput:
    return critic.run(
        query=SAMPLE_QUERY,
        contexts=SAMPLE_CONTEXTS,
        proposal=proposal_output,
        query_id="q_test",
    )


# ---------------------------------------------------------------------------
# AgentBase
# ---------------------------------------------------------------------------

class TestAgentBase:

    def test_agent_dry_run_returns_content(self, proposer):
        """Dry-run mode returns non-empty mock content."""
        response, trace = proposer.call_llm(
            system_prompt="You are a test agent.",
            user_prompt="Test prompt.",
        )
        assert isinstance(response, str)
        assert len(response) > 0
        assert trace.dry_run is True

    def test_agent_trace_required_fields(self, proposer):
        """AgentTrace has all required fields after dry-run call."""
        _, trace = proposer.call_llm("sys", "user")
        assert trace.agent_role == "proposer"
        assert trace.call_id != ""
        assert trace.timestamp > 0
        assert trace.prompt_tokens >= 0
        assert trace.completion_tokens >= 0

    def test_agent_trace_stored(self, proposer):
        """Traces accumulate on agent instance."""
        assert len(proposer.traces) == 0
        proposer.call_llm("sys", "user 1")
        proposer.call_llm("sys", "user 2")
        assert len(proposer.traces) == 2

    def test_dry_run_zero_cost_indicator(self, proposer):
        """Dry-run trace model string contains DRY-RUN marker."""
        _, trace = proposer.call_llm("sys", "user")
        assert "DRY-RUN" in trace.model

    def test_role_defaults(self, proposer, critic, judge):
        """Each agent reports correct role."""
        assert proposer.role == "proposer"
        assert critic.role == "critic"
        assert judge.role == "judge"


# ---------------------------------------------------------------------------
# Sycophancy pre-screen
# ---------------------------------------------------------------------------

class TestSycophancyPreScreen:

    def test_detects_sycophantic_opener(self, critic):
        """Certainly/Absolutely at start → sycophantic_opener flagged."""
        text = "Certainly! Here is a comprehensive answer to your question."
        flags = critic._pre_screen_sycophancy(text)
        assert "sycophantic_opener" in flags

    def test_detects_closing_offer(self, critic):
        """'Let me know if you have questions' → closing_offer flagged."""
        text = "The answer is X. Let me know if you need more information."
        flags = critic._pre_screen_sycophancy(text)
        assert "closing_offer" in flags

    def test_clean_response_no_flags(self, critic):
        """Clean technical response → no flags."""
        text = "The consolidation phase prunes low-weight edges in the knowledge graph."
        flags = critic._pre_screen_sycophancy(text)
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# ProposerAgent
# ---------------------------------------------------------------------------

class TestProposerAgent:

    def test_proposer_returns_agent_output(self, proposal_output):
        assert isinstance(proposal_output, AgentOutput)
        assert proposal_output.agent_role == "proposer"
        assert proposal_output.succeeded

    def test_proposer_content_non_empty(self, proposal_output):
        assert len(proposal_output.content) > 0

    def test_proposer_parsed_has_keys(self, proposal_output):
        parsed = proposal_output.parsed
        assert "answer" in parsed
        assert "confidence" in parsed
        assert "evidence" in parsed
        assert "reasoning" in parsed

    def test_proposer_context_formatting(self, proposer):
        contexts = ["Context A", "Context B"]
        formatted = proposer._format_contexts(contexts)
        assert "[Context 1]" in formatted
        assert "[Context 2]" in formatted
        assert "Context A" in formatted

    def test_proposer_parse_confidence_valid(self, proposer):
        """Parser extracts confidence as float."""
        text = "ANSWER: X\nCONFIDENCE: 0.85\nEVIDENCE: doc1\nREASONING: Y"
        parsed = proposer._parse_response(text)
        assert abs(parsed["confidence"] - 0.85) < 1e-9

    def test_proposer_parse_confidence_invalid(self, proposer):
        """Invalid confidence → defaults to 0.0."""
        text = "ANSWER: X\nCONFIDENCE: high\nEVIDENCE: doc1\nREASONING: Y"
        parsed = proposer._parse_response(text)
        assert parsed["confidence"] == 0.0


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class TestCriticAgent:

    def test_critic_returns_agent_output(self, critique_output):
        assert isinstance(critique_output, AgentOutput)
        assert critique_output.agent_role == "critic"
        assert critique_output.succeeded

    def test_critic_parsed_has_keys(self, critique_output):
        parsed = critique_output.parsed
        assert "reward_hacking_detected" in parsed
        assert "safety_score" in parsed
        assert "recommendation" in parsed
        assert "factual_accuracy" in parsed

    def test_critic_parse_reward_hacking_true(self, critic):
        text = "CRITIQUE: Issues found.\nREWARD_HACKING_DETECTED: True\nFACTUAL_ACCURACY: Low\nSAFETY_SCORE: 0.3\nRECALL_VERIFIED: False\nRECOMMENDATION: Reject"
        parsed = critic._parse_response(text)
        assert parsed["reward_hacking_detected"] is True
        assert parsed["recommendation"] == "Reject"

    def test_critic_parse_reward_hacking_false(self, critic):
        text = "CRITIQUE: Looks good.\nREWARD_HACKING_DETECTED: False\nFACTUAL_ACCURACY: High\nSAFETY_SCORE: 0.92\nRECALL_VERIFIED: True\nRECOMMENDATION: Accept"
        parsed = critic._parse_response(text)
        assert parsed["reward_hacking_detected"] is False
        assert parsed["recommendation"] == "Accept"
        assert abs(parsed["safety_score"] - 0.92) < 1e-9

    def test_critic_parse_hacking_indicators(self, critic):
        text = "CRITIQUE: X\nREWARD_HACKING_DETECTED: True\nHACKING_INDICATORS: sycophantic_opener, confidence_inflation\nFACTUAL_ACCURACY: Low\nSAFETY_SCORE: 0.4\nRECALL_VERIFIED: False\nRECOMMENDATION: Reject"
        parsed = critic._parse_response(text)
        assert "sycophantic_opener" in parsed["hacking_indicators"]
        assert "confidence_inflation" in parsed["hacking_indicators"]


# ---------------------------------------------------------------------------
# JudgeAgent
# ---------------------------------------------------------------------------

class TestJudgeAgent:

    def test_judge_returns_agent_output(self, judge, proposal_output, critique_output):
        output = judge.run(
            query=SAMPLE_QUERY,
            proposal=proposal_output,
            critique=critique_output,
            query_id="q_test",
        )
        assert isinstance(output, AgentOutput)
        assert output.agent_role == "judge"

    def test_judge_parsed_has_keys(self, judge, proposal_output, critique_output):
        output = judge.run(
            query=SAMPLE_QUERY,
            proposal=proposal_output,
            critique=critique_output,
        )
        assert "verdict" in output.parsed
        assert "final_safety_score" in output.parsed
        assert "final_accuracy_score" in output.parsed
        assert "reward_hacking_confirmed" in output.parsed

    def test_judge_parse_verdict_pass(self, judge):
        text = "JUDGMENT: Good.\nREWARD_HACKING_CONFIRMED: False\nFINAL_SAFETY_SCORE: 0.95\nFINAL_ACCURACY_SCORE: 0.90\nVERDICT: Pass\nREASONING: All checks passed."
        parsed = judge._parse_response(text)
        assert parsed["verdict"] == "Pass"
        assert abs(parsed["final_safety_score"] - 0.95) < 1e-9
        assert abs(parsed["final_accuracy_score"] - 0.90) < 1e-9
        assert parsed["reward_hacking_confirmed"] is False

    def test_judge_parse_verdict_fail(self, judge):
        text = "JUDGMENT: Failed.\nREWARD_HACKING_CONFIRMED: True\nFINAL_SAFETY_SCORE: 0.2\nFINAL_ACCURACY_SCORE: 0.1\nVERDICT: Fail\nREASONING: Reward hacking confirmed."
        parsed = judge._parse_response(text)
        assert parsed["verdict"] == "Fail"
        assert parsed["reward_hacking_confirmed"] is True


# ---------------------------------------------------------------------------
# DebateRound
# ---------------------------------------------------------------------------

class TestDebateRound:

    def test_debate_round_completes(self, debate_round):
        """Full A→B→C pipeline completes in dry-run mode."""
        result = debate_round.run(
            query_id="q1",
            query=SAMPLE_QUERY,
            contexts=SAMPLE_CONTEXTS,
        )
        assert isinstance(result, DebateResult)
        assert result.succeeded

    def test_debate_result_has_all_fields(self, debate_round):
        result = debate_round.run("q1", SAMPLE_QUERY, SAMPLE_CONTEXTS)
        assert result.query_id == "q1"
        assert result.n_contexts == len(SAMPLE_CONTEXTS)
        assert result.verdict in ("Pass", "Conditional", "Fail")
        assert 0.0 <= result.final_safety_score <= 1.0
        assert 0.0 <= result.final_accuracy_score <= 1.0
        assert isinstance(result.reward_hacking_confirmed, bool)
        assert result.total_latency_seconds >= 0

    def test_debate_result_serializable(self, debate_round):
        """to_dict() produces valid JSON."""
        result = debate_round.run("q1", SAMPLE_QUERY, SAMPLE_CONTEXTS)
        result_dict = result.to_dict()
        serialized = json.dumps(result_dict)
        parsed_back = json.loads(serialized)
        assert parsed_back["query_id"] == "q1"

    def test_debate_result_dry_run_flag(self, debate_round):
        result = debate_round.run("q1", SAMPLE_QUERY, SAMPLE_CONTEXTS)
        assert result.dry_run is True

    def test_debate_all_traces_present(self, debate_round):
        result = debate_round.run("q1", SAMPLE_QUERY, SAMPLE_CONTEXTS)
        traces = result.all_traces
        assert len(traces) == 3
        roles = [t.agent_role for t in traces]
        assert "proposer" in roles
        assert "critic" in roles
        assert "judge" in roles

    def test_debate_token_counts_positive(self, debate_round):
        result = debate_round.run("q1", SAMPLE_QUERY, SAMPLE_CONTEXTS)
        assert result.total_prompt_tokens >= 0
        assert result.total_completion_tokens >= 0


# ---------------------------------------------------------------------------
# Orchestration — sync mode (no Redis required)
# ---------------------------------------------------------------------------

class TestOrchestrationSync:

    def test_dispatch_sync_batch(self):
        """Sync dispatch runs debates without Celery/Redis."""
        from eval_engine.orchestration.celery_tasks import dispatch_debate_batch

        records = [
            {"query_id": f"q{i}", "query": f"Query {i}", "contexts": SAMPLE_CONTEXTS}
            for i in range(3)
        ]
        results = dispatch_debate_batch(
            records=records,
            dry_run=True,
            use_celery=False,
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, dict)

    def test_collect_sync_results(self):
        """collect_debate_results handles sync (plain dict) results."""
        from eval_engine.orchestration.celery_tasks import collect_debate_results

        fake_results = [
            {"query_id": "q1", "verdict": "Pass"},
            {"query_id": "q2", "verdict": "Fail"},
        ]
        collected = collect_debate_results(fake_results)
        assert len(collected) == 2
        assert collected[0]["query_id"] == "q1"

    def test_run_debate_task_dry_run(self):
        """run_debate_task returns serializable dict in dry-run."""
        from eval_engine.orchestration.celery_tasks import run_debate_task

        result = run_debate_task(
            query_id="q_direct",
            query=SAMPLE_QUERY,
            contexts=SAMPLE_CONTEXTS,
            dry_run=True,
        )
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "final_safety_score" in result
