"""Regression tests for the mid-stream-drop recovery cancellation fix.

On the api_server/WebUI path, a provider mid-stream transport drop arms the
text-continuation recovery but the same event tears down the browser SSE leg,
and the api_server drains the session with ``agent.interrupt("SSE client
disconnected")`` / ``("SSE task cancelled")``. Those messages mean "no client
listening", NOT user intent to stop: ``begin_iteration`` must not treat them as
a user abort, and ``apply_retry_restarts`` must let an armed recovery run even
when interrupted by the SSE reaper. A genuine user stop (carrying the user's own
text) must still win in both functions.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.turn_iteration_prep import apply_retry_restarts, begin_iteration

SSE_REAPER_MSG = "SSE client disconnected"
GENUINE_MSG = "user typed this"


def _fake_budget(consume=True, refund=False):
    return SimpleNamespace(consume=lambda **_: consume, refund=lambda **_: refund)


def _begin_agent(*, interrupt_requested, interrupt_message):
    return SimpleNamespace(
        quiet_mode=True,
        _interrupt_requested=interrupt_requested,
        _interrupt_message=interrupt_message,
        _drain_pending_redirect=lambda: None,
        _checkpoint_mgr=SimpleNamespace(new_turn=lambda: None),
        _budget_grace_call=False,
        iteration_budget=_fake_budget(consume=True),
        _api_call_count=0,
        _touch_activity=lambda *a, **k: None,
    )


def _call_begin(agent):
    return begin_iteration(
        agent,
        messages=[],
        conversation_history=None,
        original_user_message=None,
        api_call_count=0,
        interrupted=False,
        _turn_exit_reason=None,
    )


def _retry_state(*, length_continue=False, rebuilt=False, redirected=False, compressed=False):
    return SimpleNamespace(
        restart_with_redirected_messages=redirected,
        restart_with_compressed_messages=compressed,
        restart_with_rebuilt_messages=rebuilt,
        restart_with_length_continuation=length_continue,
    )


def _apply_agent(*, interrupt_requested, interrupt_message):
    return SimpleNamespace(
        quiet_mode=True,
        _interrupt_requested=interrupt_requested,
        _interrupt_message=interrupt_message,
        max_tokens=4096,
        _ephemeral_max_output_tokens=None,
        _requested_output_cap_from_api_kwargs=lambda api_kwargs: None,
        iteration_budget=_fake_budget(consume=True, refund=True),
    )


def _call_apply(agent, _retry):
    return apply_retry_restarts(
        agent,
        _retry=_retry,
        response=None,
        interrupted=agent._interrupt_requested,
        messages=[],
        conversation_history=None,
        user_message=None,
        api_kwargs={},
        current_turn_user_idx=0,
        final_response=None,
        retry_count=0,
        api_call_count=1,
        length_continue_retries=0,
        _preflight_compression_blocked=False,
        _turn_exit_reason=None,
    )


def test_begin_iteration_reaper_interrupt_does_not_break():
    """An SSE reaper interrupt means 'no client listening', not 'user stop': the
    iteration must fall through so the armed recovery can re-issue the turn."""
    agent = _begin_agent(interrupt_requested=True, interrupt_message=SSE_REAPER_MSG)
    result = _call_begin(agent)
    assert result.action == "fallthrough"
    assert result._turn_exit_reason is None


def test_begin_iteration_reaper_task_cancelled_does_not_break():
    """The sibling reaper message behaves identically to the disconnect one."""
    agent = _begin_agent(interrupt_requested=True, interrupt_message="SSE task cancelled")
    result = _call_begin(agent)
    assert result.action == "fallthrough"
    assert result._turn_exit_reason is None


def test_begin_iteration_genuine_interrupt_breaks():
    """A genuine user interrupt (carrying the user's own text) still breaks."""
    agent = _begin_agent(interrupt_requested=True, interrupt_message=GENUINE_MSG)
    result = _call_begin(agent)
    assert result.action == "break"
    assert result._turn_exit_reason == "interrupted_by_user"
    assert result.interrupted is True


def test_apply_reaper_interrupt_with_length_continuation_continues():
    """Reaper interrupt + armed length continuation: the recovery must run, not break."""
    agent = _apply_agent(interrupt_requested=True, interrupt_message=SSE_REAPER_MSG)
    _retry = _retry_state(length_continue=True)
    result = _call_apply(agent, _retry)
    assert result.action == "continue"
    assert result._turn_exit_reason is None


def test_apply_reaper_interrupt_with_rebuilt_messages_continues():
    """Reaper interrupt + armed rebuilt-messages fallback: the recovery must run."""
    agent = _apply_agent(interrupt_requested=True, interrupt_message=SSE_REAPER_MSG)
    _retry = _retry_state(rebuilt=True)
    result = _call_apply(agent, _retry)
    assert result.action == "continue"
    assert result._turn_exit_reason is None


def test_apply_reaper_interrupt_no_recovery_armed_breaks():
    """Reaper interrupt with no recovery armed: no restart to honor, so break."""
    agent = _apply_agent(interrupt_requested=True, interrupt_message=SSE_REAPER_MSG)
    _retry = _retry_state()
    result = _call_apply(agent, _retry)
    assert result.action == "break"
    assert result._turn_exit_reason == "interrupted_during_api_call"


def test_apply_genuine_interrupt_wins_even_with_recovery_armed():
    """A genuine user stop still wins even when a recovery is armed."""
    agent = _apply_agent(interrupt_requested=True, interrupt_message=GENUINE_MSG)
    _retry = _retry_state(length_continue=True)
    result = _call_apply(agent, _retry)
    assert result.action == "break"
    assert result._turn_exit_reason == "interrupted_during_api_call"
