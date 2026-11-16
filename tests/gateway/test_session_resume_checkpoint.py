"""S3 resume-checkpoints persist per-session and survive a simulated restart.

Continue=Resume rehydrates the persisted checkpointer snapshot of the interrupted tool round
(LangGraph-checkpointer analog). This test proves the SessionStore layer that carries it:
  - set/get round-trips the resumable record
  - clear removes it (and set with None clears)
  - a simulated restart (a brand-new SessionStore instance reading the same sessions dir)
    still sees it — the very thing "continue" after a gateway restart needs
  - the sanitizer drops unknown/secret-ish/replayable-CoTHazard keys and refuses hollow rows
  - api_key / raw reasoning blobs beyond the allowlist are NEVER serialized
"""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionSource,
    SessionStore,
    sanitize_resume_checkpoint,
)

CHECKPOINT = {
    "kind": "tool_round",
    "turn_id": "turn_abc123",
    "round_start": 2,
    "tool_calls": [
        {"name": "search_files", "arguments": {"pattern": "*.py"}},
        {"name": "read_file", "arguments": {"path": "/tmp/x.py"}},
    ],
    "partial_reasoning": "was about to search then read the file",
    "created_at": "2026-11-16T00:00:00",
}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def test_checkpoint_persists_and_survives_restart(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    store.set_resume_checkpoint(session_key, CHECKPOINT)
    assert store.get_resume_checkpoint(session_key) == {
        "kind": "tool_round",
        "turn_id": "turn_abc123",
        "round_start": 2,
        "tool_calls": [
            {"name": "search_files", "arguments": {"pattern": "*.py"}},
            {"name": "read_file", "arguments": {"path": "/tmp/x.py"}},
        ],
        "partial_reasoning": "was about to search then read the file",
        "created_at": "2026-11-16T00:00:00",
    }

    # Simulated restart: a brand-new store instance reads the same dir.
    store2 = store_factory()
    assert store2.get_resume_checkpoint(session_key) == store.get_resume_checkpoint(session_key)
    # A different SESSION does NOT inherit it (session-keyed, like LangGraph thread_id).
    other_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u2",
        chat_id="c2",
        user_name="other",
        chat_type="dm",
    )
    other_key = store.get_or_create_session(other_source).session_key
    assert other_key != session_key
    assert store2.get_resume_checkpoint(other_key) is None


def test_checkpoint_clear(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    store.set_resume_checkpoint(session_key, CHECKPOINT)
    assert store.clear_resume_checkpoint(session_key) is True
    assert store.get_resume_checkpoint(session_key) is None
    # Clearing again is a no-op (idempotent).
    assert store.clear_resume_checkpoint(session_key) is False

    # set with None also clears.
    store.set_resume_checkpoint(session_key, CHECKPOINT)
    store.set_resume_checkpoint(session_key, None)
    assert store.get_resume_checkpoint(session_key) is None


def test_sanitize_resume_checkpoint():
    # None / empty / wrong type -> hollow, nothing resumable.
    assert sanitize_resume_checkpoint(None) is None
    assert sanitize_resume_checkpoint({}) is None
    assert sanitize_resume_checkpoint("x") is None
    # No pending tool calls -> nothing resumable.
    assert sanitize_resume_checkpoint({"tool_calls": []}) is None
    assert sanitize_resume_checkpoint({"kind": "tool_round"}) is None

    # Unknown / secret / replayable-hazard keys are dropped; only allowlist kept, no CoT-as-content.
    noisy = {
        "tool_calls": [{"name": "bash", "arguments": "ls", "api_key": "sk-secret"}],
        "api_key": "sk-leak",
        "raw_reasoning_blob": "chain of thought that must not persist",
        "provider_secret": "nope",
    }
    cleaned = sanitize_resume_checkpoint(noisy)
    assert cleaned == {"kind": "tool_round", "tool_calls": [{"name": "bash", "arguments": "ls"}]}
    assert "api_key" not in cleaned
    assert "raw_reasoning_blob" not in cleaned
    assert "provider_secret" not in cleaned
