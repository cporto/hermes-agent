"""A paid provider's out-of-credits error must ALERT the user even when a
working fallback covers the turn (# cost rule: 402, and OpenAI-style 429
insufficient_quota, from a PAID provider -> alert the user even if the
fallback succeeds).

The billing-failover notification path (_log_fallback_activated) is driven by
real classify_api_error verdicts; we assert BOTH:
  (a) classification: HTTP 402 and 429 "insufficient_quota" are billing (fallback armed), and
  (b) delivery: a billing failover covered by a fallback emits a USER-VISIBLE
      warning on the agent (not just a log line), while a non-billing reason stays quiet.

Deterministic + mocked: fake error objects and a fake agent with a captured
_emit_warning. No real API calls, no spend.
"""
from agent.chat_completion_helpers import _log_fallback_activated
from agent.error_classifier import FailoverReason, classify_api_error


class _StatusError(Exception):
    def __init__(self, message, status_code, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}


def _billing_402():
    return _StatusError(
        "Error code: 402 - This request requires more credits, or fewer max_tokens. "
        "You requested up to 65536 tokens, but can only afford 56272.",
        402,
    )


def _billing_429_insufficient_quota():
    return _StatusError(
        "Error code: 429 - {'error': {'code': 'insufficient_quota', "
        "'message': 'You exceeded your current quota.'}}",
        429,
    )


class _UserAwareAgent:
    """Fake agent: captures anything user-visible emitted during the turn."""

    def __init__(self):
        self.warnings = []

    def _emit_warning(self, message):
        self.warnings.append(str(message))


def test_402_and_429_insufficient_quota_classify_as_billing():
    assert classify_api_error(_billing_402(), provider="openrouter").reason == FailoverReason.billing
    assert classify_api_error(_billing_402(), provider="openrouter").should_fallback
    assert classify_api_error(_billing_429_insufficient_quota(), provider="openai").reason == FailoverReason.billing
    assert classify_api_error(_billing_429_insufficient_quota(), provider="openai").should_fallback


def test_billing_fallback_alerts_user_even_when_covered():
    # Fallback ACTIVATED (turn continues) but the primary was out of credits:
    # the user must still be alerted that a PAID provider is out of credits.
    agent = _UserAwareAgent()
    _log_fallback_activated(agent, FailoverReason.billing, "paid/model", "openrouter", "fb/model", "local")
    assert len(agent.warnings) == 1, "covered billing failover must emit exactly one user-visible warning"
    msg = agent.warnings[0].lower()
    assert "credit" in msg or "quota" in msg or "billing" in msg, f"alert should name the cause: {msg}"
    assert "paid/model" in msg and "fb/model" in msg, f"alert should name both the failing model and the fallback: {msg}"


def test_non_billing_reason_stays_silent():
    # A plain server_error failover is routine; it must NOT fake a billing alert.
    agent = _UserAwareAgent()
    _log_fallback_activated(agent, FailoverReason.server_error, "a", "p", "b", "q")
    assert agent.warnings == []


def test_none_agent_is_safe():
    # The legacy call path passes agent=None (see test_credit_404_fallback); must not crash.
    _log_fallback_activated(None, FailoverReason.billing, "a", "p", "b", "q")