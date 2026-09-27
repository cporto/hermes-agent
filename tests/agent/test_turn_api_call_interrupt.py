"""Regression tests for the SSE reaper interrupt at the ``handle_api_interrupt`` gate.

On the api_server/WebUI path, tearing down the browser SSE leg (tab switch / focus
loss on mobile Safari included) fires ``agent.interrupt("SSE client disconnected" /
"SSE task cancelled")``. Those reaper messages mean "no client listening", NOT user
intent to stop. When the drop happens before anything streamed (a 0.0s interrupt,
nothing to length-continue, no rebuilt fallback), ``handle_api_interrupt`` used to
latch ``interrupted=True`` and fabricate an "Operation interrupted: waiting for model
response (0.0s elapsed)" final response, ending the turn as a dead-end.

The fix treats that as a transport-only reaper with no armed recovery: clear the
interrupt and re-issue the same logical iteration (``restart_with_redirected_messages``),
keeping ``interrupted`` False so the re-issued turn finalizes as a real response.

A genuine user stop (non-reaper message) is unaffected. A reaper interrupt WITH an
armed length-continuation/fallback recovery is left to that existing recovery path.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.turn_api_call import handle_api_interrupt

SSE_REAPER_MSG = "SSE client disconnected"
SSE_TASK_CANCELLED = "SSE task cancelled"
GENUINE_MSG = "user typed this"


def _retry(*, length_continue=False, rebuilt=False, redirected=False):
    return SimpleNamespace(
        restart_with_length_continuation=length_continue,
        restart_with_rebuilt_messages=rebuilt,
        restart_with_redirected_messages=redirected,
    )


def _agent(*, interrupt_message, current_streamed_text=""):
    calls = {"cleared": 0, "persisted": 0}

    def _clear_interrupt(*, preserve_redirect=False):
        calls["cleared"] += 1
        return True

    def _persist_session(*_a, **_k):
        calls["persisted"] += 1

    return SimpleNamespace(
        _interrupt_message=interrupt_message,
        _has_pending_redirect=lambda: False,
        clear_interrupt=_clear_interrupt,
        _vprint=lambda *a, **k: None,
        _strip_think_blocks=lambda t: t,
        _current_streamed_assistant_text=current_streamed_text,
        log_prefix="",
        thinking_callback=None,
        _persist_session=_persist_session,
        _calls=calls,
    )


def _call(agent, r=None, interrupted=False):
    return handle_api_interrupt(
        agent,
        _retry=r or _retry(),
        thinking_spinner=None,
        messages=[],
        conversation_history=[],
        api_start_time=0.0,
        interrupted=interrupted,
        final_response=None,
    )


def test_reaper_0s_no_recovery_reissues_and_does_not_flag_interrupted():
    """A 0.0s SSE reaper interrupt with no partial output / no armed recovery must
    re-issue the same logical iteration (continue via redirect restart) and keep
    ``interrupted`` False — NOT end the turn with 'Operation interrupted'."""
    agent = _agent(interrupt_message=SSE_REAPER_MSG)
    r = _retry()
    verdict = _call(agent, r=r)
    assert verdict.action == "break"  # leaves the retry loop; outer rebuilds
    assert verdict.interrupted is False  # do NOT latch interrupted=True
    assert r.restart_with_redirected_messages is True  # re-issue same iteration
    assert agent._calls["cleared"] == 1  # transport-only interrupt consumed


def test_reaper_task_cancelled_reissues_like_disconnect():
    """The sibling reaper message behaves identically to the disconnect one."""
    agent = _agent(interrupt_message=SSE_TASK_CANCELLED)
    r = _retry()
    verdict = _call(agent, r=r)
    assert verdict.interrupted is False
    assert r.restart_with_redirected_messages is True


def test_reaper_with_armed_recovery_left_to_existing_path():
    """A reaper interrupt WITH an armed recovery must fall through to that recovery
    (the length-continuation / rebuilt-messages path), not be re-issued via redirect."""
    agent = _agent(interrupt_message=SSE_REAPER_MSG)
    r = _retry(length_continue=True)
    verdict = _call(agent, r=r)
    # Leaves the retry loop and latches interrupted (existing handle path); the
    # apply_retry_restarts gate then runs the armed recovery.
    assert verdict.interrupted is True
    assert r.restart_with_redirected_messages is False


def test_genuine_user_stop_still_flags_interrupted():
    """A genuine user stop (non-reaper message) still ends the turn interrupted."""
    agent = _agent(interrupt_message=GENUINE_MSG)
    r = _retry()
    verdict = _call(agent, r=r)
    assert verdict.interrupted is True
    assert r.restart_with_redirected_messages is False
    assert agent._calls["cleared"] == 0
