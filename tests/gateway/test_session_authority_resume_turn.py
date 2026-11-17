"""S3 Continue=Resume via the LIVE authority seam (TurnRunner._run_conversation_with_approval).

The S3 continue=resume feature rehydrates a persisted interrupt checkpoint on the authority path:
- READ side turns a bare "continue" API admission into a scripted tool-round prelude passed to
  run_conversation (and marks the turn as resumed).
- WRITE side persists the agent's stamped checkpoint for API admissions that are NOT a resume.
- CLEAR side clears the consumed checkpoint after a resumed turn.
- a fail-safe: no checkpoint + continue, or a checkpoint but non-continue, does NOT resume and does
  NOT disturb the store.

This drives the exact seam gateway/run_turn_runner.py::_run_conversation_with_approval wires, using
the real SessionStore keyed by the route (the authority's per-session route is ctx.session_key here,
matching the production read/write/clear keys). Pure in-process, no LLM, no sleeps.
"""
import pytest

from gateway.session_api_turn import (
    admit_api_turn,
    api_execution,
    prepare_api_execution,
)
from gateway.session_ingress import execute_admission
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


CHECKPOINT = {
    "kind": "tool_round",
    "round_start": 2,
    "tool_calls": [
        {"name": "search_files", "arguments": {"pattern": "*.py"}},
        {"name": "read_file", "arguments": {"path": "/tmp/x.py"}},
    ],
}


def _admit_and_turn(owner, ref, route, message, agent):
    """Run the authority seam for one API turn. Returns the seam's result dict."""
    from types import SimpleNamespace

    ctx = TurnContext(message=message, session_id=ref.session_id, session_key=route,
                      source=SimpleNamespace(user_id='u', user_name='U', is_bot=False))
    turn = TurnRunner(
        SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: [],
                        session_store=owner.runner.session_store),
        ctx,
    )
    payload = {'text': message}
    token = api_execution.set(prepare_api_execution(owner, ref, payload))
    try:
        return turn._run_conversation_with_approval(agent, [], None, None, None)
    finally:
        api_execution.reset(token)


def _make_agent(run_conversation, pending_checkpoint=None):
    """Fake agent in the shape the seam expects (duck-typed run_conversation + optional stamp)."""
    from types import SimpleNamespace

    agent = SimpleNamespace()
    agent.run_conversation = run_conversation
    if pending_checkpoint is not None:
        agent._s3_pending_resume_checkpoint = pending_checkpoint
    return agent


async def _admit(api, owner, session_id, message):
    """Admit an API turn, drive admission so the route/session-key exists, return (ref, route)."""
    _, ref, row = admit_api_turn(api, session_id=session_id, user_message=message,
                                 conversation_history=[])
    from gateway.session_results import execution_result

    async def handle(event):
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'
    owner.runner._handle_message = handle
    await execute_admission(owner, ref, row)
    route = owner.sessions[ref.session_id].route
    assert route is not None
    # The admission must register the route in the real SessionStore, else set/get are no-ops.
    assert route in owner.runner.session_store._entries
    return ref, route


@pytest.mark.asyncio
async def test_read_side_delivers_prelude(api, owner, tmp_path, monkeypatch):
    """Case 1: READ side — a seeded checkpoint + 'continue' feeds a prelude and marks resumed."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ref, route = await _admit(api, owner, 'resume-s', 'continue')
    store = owner.runner.session_store
    store.set_resume_checkpoint(route, CHECKPOINT)
    assert store.get_resume_checkpoint(route) == CHECKPOINT

    seen = {}

    def run(message, *, prelude=None, **kwargs):
        seen['prelude'] = prelude
        return {'final_response': 'ok'}
    agent = _make_agent(run)
    result = _admit_and_turn(owner, ref, route, 'continue', agent)
    assert seen['prelude'] is not None
    assert result.get('resumed_from_checkpoint') is True


@pytest.mark.asyncio
async def test_write_side_persists_on_interrupt(api, owner, tmp_path, monkeypatch):
    """Case 2: WRITE side — a non-continue API turn with a stamped checkpoint persists it to the
    route's key."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ref, route = await _admit(api, owner, 'write-s', 'hi')
    store = owner.runner.session_store
    assert store.get_resume_checkpoint(route) is None

    seen = {}

    def run(message, *, prelude=None, **kwargs):
        seen['prelude'] = prelude
        return {'final_response': 'ok'}
    agent = _make_agent(run, pending_checkpoint=CHECKPOINT)
    result = _admit_and_turn(owner, ref, route, 'hi', agent)
    assert seen['prelude'] is None  # not a continue -> no read-side prelude
    assert result.get('resumed_from_checkpoint') is None
    assert store.get_resume_checkpoint(route) == CHECKPOINT


@pytest.mark.asyncio
async def test_clear_side_clears_after_resume(api, owner, tmp_path, monkeypatch):
    """Case 3: CLEAR side — a consumed resume clears the persisted checkpoint from the route."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ref, route = await _admit(api, owner, 'clear-s', 'continue')
    store = owner.runner.session_store
    store.set_resume_checkpoint(route, CHECKPOINT)
    assert store.get_resume_checkpoint(route) == CHECKPOINT

    def run(message, *, prelude=None, **kwargs):
        return {'final_response': 'ok'}
    agent = _make_agent(run)
    result = _admit_and_turn(owner, ref, route, 'continue', agent)
    assert result.get('resumed_from_checkpoint') is True
    assert store.get_resume_checkpoint(route) is None


@pytest.mark.asyncio
async def test_negative_no_checkpoint_does_not_resume(api, owner, tmp_path, monkeypatch):
    """Case 4a: fail-safe — 'continue' with nothing in the store -> no prelude, no resumed flag."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ref, route = await _admit(api, owner, 'neg1-s', 'continue')
    store = owner.runner.session_store
    assert store.get_resume_checkpoint(route) is None

    seen = {}

    def run(message, *, prelude=None, **kwargs):
        seen['prelude'] = prelude
        return {'final_response': 'ok'}
    agent = _make_agent(run)
    result = _admit_and_turn(owner, ref, route, 'continue', agent)
    assert seen['prelude'] is None
    assert 'resumed_from_checkpoint' not in result
    assert store.get_resume_checkpoint(route) is None


@pytest.mark.asyncio
async def test_negative_non_continue_preserves_checkpoint(api, owner, tmp_path, monkeypatch):
    """Case 4b: fail-safe — a checkpoint present but a non-continue message leaves it untouched."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ref, route = await _admit(api, owner, 'neg2-s', 'hello')
    store = owner.runner.session_store
    store.set_resume_checkpoint(route, CHECKPOINT)
    assert store.get_resume_checkpoint(route) == CHECKPOINT

    seen = {}

    def run(message, *, prelude=None, **kwargs):
        seen['prelude'] = prelude
        return {'final_response': 'ok'}
    agent = _make_agent(run)
    result = _admit_and_turn(owner, ref, route, 'hello', agent)
    assert seen['prelude'] is None
    assert 'resumed_from_checkpoint' not in result
    # Untouched: neither cleared nor overwritten.
    assert store.get_resume_checkpoint(route) == CHECKPOINT


@pytest.mark.asyncio
async def test_write_guard_no_double_persist_during_resume(api, owner, tmp_path, monkeypatch):
    """Case 5: guard — during a resumed turn the write side must NOT re-persist the (consumed)
    checkpoint; the seeded checkpoint is cleared, not replaced."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ref, route = await _admit(api, owner, 'g5-s', 'continue')
    store = owner.runner.session_store
    store.set_resume_checkpoint(route, CHECKPOINT)
    assert store.get_resume_checkpoint(route) == CHECKPOINT

    def run(message, *, prelude=None, **kwargs):
        return {'final_response': 'ok'}
    agent = _make_agent(run, pending_checkpoint=CHECKPOINT)  # stamped AND a seeded resume
    result = _admit_and_turn(owner, ref, route, 'continue', agent)
    assert result.get('resumed_from_checkpoint') is True
    # The consumed resume cleared the seeded checkpoint; the pending stamp was NOT re-persisted
    # (write side guarded on _resumed_s3), so nothing remains on the route's key.
    assert store.get_resume_checkpoint(route) is None
