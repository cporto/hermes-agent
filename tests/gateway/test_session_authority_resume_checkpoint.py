"""S3 resume-checkpoint persistence on stop (gateway.session_authority).

When the user stops a turn mid-tool-round, the running agent stamps the pending (unexecuted)
tool calls; ``authority.interrupt`` persists that stamp to the durable, session-keyed store so a
later "continue" rehydrates and RE-RUNS the same tool call (the Continue=Resume half of the
Stop/Continue pair, LangGraph-checkpointer analog). This tests the bridge:

  - a stamped agent checkpoint is persisted to the session store keyed by session
  - the in-memory stamp is consumed (one-shot) so a second stop can't double-write
  - no stamp -> nothing persisted; missing store / setter -> no raise (fail-safe)
"""
import pytest

from gateway.session_authority import _persist_resume_checkpoint_on_interrupt

CHECKPOINT = {
    "kind": "tool_round",
    "round_start": 2,
    "tool_calls": [
        {"name": "search_files", "arguments": {"pattern": "*.py"}},
        {"name": "read_file", "arguments": {"path": "/tmp/x.py"}},
    ],
}


class _Agent:
    def __init__(self, checkpoint):
        self._s3_pending_resume_checkpoint = checkpoint


class _Store:
    def __init__(self):
        self.written = {}

    def set_resume_checkpoint(self, key, checkpoint):
        self.written[key] = checkpoint


class _Runner:
    def __init__(self, store=None):
        self.session_store = store


class _Authority:
    def __init__(self, runner):
        self.runner = runner


def test_persists_stamped_checkpoint_keyed_by_session():
    store = _Store()
    authority = _Authority(_Runner(store))
    _persist_resume_checkpoint_on_interrupt(authority, _Agent(CHECKPOINT), "agent:main:tg:c1")
    assert store.written == {"agent:main:tg:c1": CHECKPOINT}


def test_stamp_consumed_after_persist():
    agent, store = _Agent(CHECKPOINT), _Store()
    _persist_resume_checkpoint_on_interrupt(_Authority(_Runner(store)), agent, "sk")
    assert getattr(agent, "_s3_pending_resume_checkpoint", None) is None


def test_no_stamp_nothing_persisted():
    agent, store = _Agent(None), _Store()
    _persist_resume_checkpoint_on_interrupt(_Authority(_Runner(store)), agent, "sk")
    assert store.written == {}


def test_missing_store_is_fail_safe():
    # No runner / no session_store -> no raise, nothing written.
    _persist_resume_checkpoint_on_interrupt(_Authority(None), _Agent(CHECKPOINT), "sk")
    authority = _Authority(_Runner(None))
    _persist_resume_checkpoint_on_interrupt(authority, _Agent(CHECKPOINT), "sk")


def test_store_without_setter_is_fail_safe():
    authority = _Authority(_Runner(object()))  # runner.session_store has no set_resume_checkpoint
    _persist_resume_checkpoint_on_interrupt(authority, _Agent(CHECKPOINT), "sk")
