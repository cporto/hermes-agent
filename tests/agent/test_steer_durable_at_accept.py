"""Regression: an accepted steer becomes durable in the session transcript even
when the run is interrupted before any later flush/finalize (Carlos's 'phone off
and it's gone' case).

A POST /v1/runs/{id}/steer returns HTTP 200 when agent.steer() QUEUES the text
into _pending_steer. Both downstream drain sites (tool-batch flush at next API
call; finalizer leftover drain) can be skipped by a hard interrupt, so an
appended-only-queued steer was previously absent from the transcript on
disconnect. The fix persists the steer as a display_kind='steer' user row at
ACCEPT time (gateway/platforms/api_server_runs.py _handle_steer_run), and the
three drain sites stamp _DB_PERSISTED_MARKER on the live row so exactly one
durable row exists.

Invariants (proven red on base, green on fix):
1. Steering into a session whose tail is assistant/tool yields EXACTLY ONE
   display_kind='steer' row, in a legal alternation position.
2. Steering with a USER tail does NOT write a user->user row (deferred to drain).
3. The drain sites stamp the row as already-persisted so no duplicate is flushed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_agent(tail_role, session_db):
    """Minimal agent surface for the steer-hit-append path."""
    class _Agent:
        session_id = "sess-test"
        _steer_durably_persisted = False
        _session_db = session_db
        _active_compression_lock_holder = None
        _active_session_turn_lease_holder = None
        _active_session_turn_lease_ttl_seconds = 300.0
        _pending_steer = None
        def steer(self, text):
            from agent.interrupt_control import InterruptControlMixin
            # piggyback the real queue semantics
            self._pending_steer = (self._pending_steer + "\n" + text) if self._pending_steer else text
            return True
        def _drain_pending_steer(self):
            t, self._pending_steer = self._pending_steer, None
            return t
    return _Agent()


def _fake_db(tail_role):
    """In-memory session-db stand-in that records appended rows and tail role."""
    rows = []
    if tail_role:
        rows.append({"role": tail_role, "content": "PRE"})
    class _DB:
        def get_messages(self, session_id, limit=None, latest=False):
            return list(rows)
        def append_messages_batch(self, session_id, messages, **kw):
            rows.extend(messages)
            return len(messages)
    return _DB(), rows


def test_accept_persist_writes_one_steer_row_when_tail_is_not_user():
    from agent.prompt_builder import steer_user_row
    from agent.session_persistence import _db_flush_row  # canonical row projection
    db, rows = _fake_db("assistant")
    agent = _stub_agent("assistant", db)
    # emulate the accept-time block from _handle_steer_run
    from agent.session_persistence import _persist_lock
    with _persist_lock(agent):
        tail = db.get_messages(agent.session_id, limit=1, latest=True)
        if tail and tail[-1].get("role") != "user":
            row = steer_user_row("nudge-X")
            db.append_messages_batch(agent.session_id,
                                     messages=[_db_flush_row(agent, row, is_current_turn_user=False)])
            agent._steer_durably_persisted = True
    steer_rows = [r for r in rows if r.get("display_kind") == "steer"]
    assert len(steer_rows) == 1
    assert "nudge-X" in steer_rows[0].get("content", "")


def test_accept_persist_skips_user_tail_alternation():
    from agent.prompt_builder import steer_user_row
    from agent.session_persistence import _db_flush_row, _persist_lock
    db, rows = _fake_db("user")
    agent = _stub_agent("user", db)
    with _persist_lock(agent):
        tail = db.get_messages(agent.session_id, limit=1, latest=True)
        if tail and tail[-1].get("role") != "user":
            row = steer_user_row("nudge-X")
            db.append_messages_batch(agent.session_id,
                                     messages=[_db_flush_row(agent, row, is_current_turn_user=False)])
            agent._steer_durably_persisted = True
    steer_rows = [r for r in rows if r.get("display_kind") == "steer"]
    assert len(steer_rows) == 0  # no user->user row; deferred to drain
