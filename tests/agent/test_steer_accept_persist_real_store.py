"""Regression: the accept-time steer persist must work against the REAL storage handle.

``_persist_pending_steer_durably`` once called ``_session_db._read_one(...)`` (a SessionDB-era raw-SQL read) and the
SessionDB-era ``set_user_message_content`` / ``append_messages_batch`` verbs. The storage handle exposes neither, so every
``/steer`` logged ``accept-time steer persist failed: AttributeError: _read_one`` and the steer was never made durable.
The older test in test_steer_durable_at_accept.py drives a fake DB, which is why it never saw this.
"""
import logging

import pytest

from agent.interrupt_control import InterruptControlMixin
from persistence.conversation.messages import TranscriptRead
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle  # noqa: F401 (fixture)


pytestmark = pytest.mark.usefixtures("close_storage_handles")


class _Agent(InterruptControlMixin):
    session_id = "sess-steer-real"

    def __init__(self, store):
        self.persistence = store
        self._session_db = store
        self._pending_steer = None
        self._pending_steer_uid = None
        self._steer_durably_persisted = False
        self._active_compression_lock_holder = None
        self._active_session_turn_lease_holder = None
        self._active_session_turn_lease_ttl_seconds = 300.0


def _steer_rows(store):
    return [r for r in transcript_rows(store, "sess-steer-real", TranscriptRead()) if r.get("display_kind") == "steer"]


def _agent(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "sess-steer-real", "api_server")
    return store, _Agent(store)


def test_accept_time_steer_persist_writes_one_row_and_does_not_log_error(tmp_path, caplog):
    store, agent = _agent(tmp_path)
    with caplog.at_level(logging.ERROR):
        assert agent.steer("nudge-A") is True
    assert not [r for r in caplog.records if "steer persist failed" in r.getMessage()], caplog.text
    rows = _steer_rows(store)
    assert len(rows) == 1 and "nudge-A" in str(rows[0]["content"])
    assert agent._steer_durably_persisted is True


def test_second_steer_extends_the_same_durable_row(tmp_path, caplog):
    store, agent = _agent(tmp_path)
    with caplog.at_level(logging.ERROR):
        agent.steer("first")
        agent.steer("second")
    assert not [r for r in caplog.records if "steer persist failed" in r.getMessage()], caplog.text
    rows = _steer_rows(store)
    assert len(rows) == 1, rows
    assert "first" in str(rows[0]["content"]) and "second" in str(rows[0]["content"])
