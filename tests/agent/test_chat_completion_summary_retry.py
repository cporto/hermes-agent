"""Regression tests for the iteration-limit summary path resilience.

Locks in the behavior added for the "couldn't summarize / 429 engine_overloaded" bug:
(1) a transient provider error (HTTP 429 / engine_overloaded / RateLimitError) on the final
summary call is RETRIED with backoff instead of surfacing the raw error, and (2) after retries
exhaust on a still-transient error, ONE contained non-mutating fallback summary attempt runs.
These are behaviour contracts, not snapshots.
"""

import types
from unittest.mock import MagicMock, patch

import pytest

from agent import chat_completion_helpers as cch


class _RateLimitError(Exception):
    """Duck-typed stand-in matching the RateLimitError class-name the classifier detects."""


def _make_agent(**overrides):
    agent = types.SimpleNamespace(
        max_iterations=60,
        suppress_status_output=True,
        quiet_mode=True,
        api_mode="chat_completions",
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        provider="custom",
        base_url="https://api.deepinfra.com/v1/openai",
        _api_max_retries=3,
        _fallback_chain=[
            {"provider": "custom", "model": "accounts/fireworks/models/deepseek-v4-flash-0731",
             "base_url": "https://api.fireworks.ai/inference/v1"},
            {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"},
            {"provider": "custom", "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
             "base_url": "https://api.deepinfra.com/v1/openai"},
        ],
    )
    agent._safe_print = MagicMock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


@pytest.fixture
def _noop_seams():
    """Neutralize the summary-path seams so only the retry/fallback wiring is under test.

    NOTE: handle_max_iterations does `from agent import relay_llm` *inside* the function, so the
    module-under-test has no `relay_llm` attribute — patch at the real module binding
    (`agent.relay_llm.complete_logical_call`), not `patch.object(cch, ...)` (would raise).
    """
    with patch.object(cch, "_iteration_summary_api_messages", return_value=[]):
        with patch("agent.relay_llm.complete_logical_call", return_value=None):
            yield


class TestIsTransientSummaryError:
    def test_rate_limit_error_class_is_transient(self):
        assert cch._is_transient_summary_error(_RateLimitError("Model busy, retry later")) is True

    def test_429_engine_overloaded_text_is_transient(self):
        assert cch._is_transient_summary_error(Exception("Error code: 429 - engine_overloaded")) is True

    def test_model_busy_retry_later_too_many_requests_transient(self):
        assert cch._is_transient_summary_error(Exception("Model busy, retry later")) is True
        assert cch._is_transient_summary_error(Exception("too many requests")) is True
        assert cch._is_transient_summary_error(Exception("rate limited")) is True

    def test_auth_and_bad_request_are_not_transient(self):
        assert cch._is_transient_summary_error(Exception("Error code: 401 - unauthorized")) is False
        assert cch._is_transient_summary_error(Exception("Error code: 400 - Extra inputs not permitted")) is False
        assert cch._is_transient_summary_error(Exception("some deterministic failure")) is False


class TestSummaryRetriesOnTransientError:
    def test_transient_429_is_retried_and_succeeds(self, _noop_seams):
        agent = _make_agent()
        calls = {"n": 0}

        def flaky_attempt(retry_count):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimitError("Error code: 429 - engine_overloaded")
            return "recovered summary"

        with patch.object(cch, "_chat_summary_attempt", return_value=flaky_attempt):
            result = cch.handle_max_iterations(agent, [], 60)

        assert calls["n"] == 2, "expect exactly one retry after the transient 429"
        assert result == "recovered summary"
        assert "couldn't summarize" not in result

    def test_transient_429_exhausting_retries_tries_fallback(self, _noop_seams):
        agent = _make_agent()
        calls = {"n": 0}

        def always_overloaded(retry_count):
            calls["n"] += 1
            raise _RateLimitError("Error code: 429 - engine_overloaded")

        with patch.object(cch, "_chat_summary_attempt", return_value=always_overloaded):
            with patch.object(cch, "_summary_fallback_attempt", return_value="summary from fallback provider") as fb:
                result = cch.handle_max_iterations(agent, [], 60)

        assert calls["n"] == agent._api_max_retries, "primary retries exhausted"
        assert fb.called, "fallback attempted after transient retries exhausted"
        assert result == "summary from fallback provider"

    def test_nontransient_error_does_not_fallback(self, _noop_seams):
        agent = _make_agent()

        def deterministic_failure(retry_count):
            raise Exception("Error code: 400 - Extra inputs not permitted")

        with patch.object(cch, "_chat_summary_attempt", return_value=deterministic_failure):
            with patch.object(cch, "_summary_fallback_attempt", return_value="fallback") as fb:
                result = cch.handle_max_iterations(agent, [], 60)

        assert not fb.called, "a deterministic 400 must NOT trigger fallback"
        # Merged production returns the max_iterations_no_summary site copy (summary
        # failed non-transiently), not the older "couldn't summarize" wording.
        assert "couldn't produce a summary" in result


class TestSummaryFallbackIsContained:
    def test_fallback_attempt_does_not_mutate_agent_model(self, _noop_seams):
        # The fallback helper must not reassign the session's model/provider/base_url.
        import inspect
        src = inspect.getsource(cch._summary_fallback_attempt)
        assert "agent.model =" not in src
        assert "agent.provider =" not in src
        assert "agent.base_url =" not in src
