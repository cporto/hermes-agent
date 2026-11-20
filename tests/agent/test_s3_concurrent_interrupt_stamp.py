"""S3 continue=resume — regression: the CONCURRENT/segmented executor must stamp a resume
checkpoint for UN-STARTED tool calls when a user interrupt lands mid-batch, and must NEVER
include a started/completed call (the microsoft/agent-framework #1695 hazard).

RED on the pre-fix tree (no stamp in _append_batch_results), GREEN after the fix.
Deterministic (no live LLM, no sleeps): we drive `_append_batch_results` directly with a
batch whose slots #2/#3 are un-started and interrupt-flagged.
"""
import types

import pytest

from agent import tool_executor as te


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _Parsed:
    """Duck-typed stand-in for te._ParsedCall (the fields _append_batch_results touches)."""

    def __init__(self, name, arguments):
        self.tool_call = _ToolCall(name, arguments)
        self.name = name
        self.args = arguments
        self.parse_error = None
        self.scope_block = None
        self.middleware_trace = []

    def ref(self, task_id):
        return types.SimpleNamespace(emit_post=lambda *a, **k: None)


class _Done:
    def __init__(self, parsed):
        self.ref = parsed
        self.result = "ok"
        self.duration = 0.1
        self.is_error = False
        self.blocked = False
        self.data_use = frozenset()  # read by upstream's _append_batch_results


class _Agent:
    def __init__(self):
        self._s3_pending_resume_checkpoint = None
        self._interrupt_requested = True  # a stop landed mid-batch
        self._vprint = lambda *a, **k: None
        self._safe_print = lambda *a, **k: None
        self.log_prefix = ""
        self.time = 0.0

    def _should_emit_quiet_tool_messages(self):
        return False


def _noop(*a, **k):
    return None


def _commit_noop(agent, messages, ref, function_result, **k):
    return ("persisted", "display", None)


def _mk_batch(completed_name, unstarted_names, done_names=(), timed_out=()):
    import functools

    def parsed(name):
        if isinstance(name, str):
            return _Parsed(name, {"_p": name})
        return name

    calls = [parsed(n) for n in completed_name + unstarted_names]
    results = []
    for i, c in enumerate(calls):
        results.append(_Done(c) if c.name in done_names else None)
    return calls, results


@pytest.fixture
def _env(monkeypatch):
    monkeypatch.setattr(te, "_unfinished_tool_result",
                        lambda agent, ref, timed_out, timeout_s: ("[cancelled]", 0.0, None))
    monkeypatch.setattr(te, "_commit_tool_result", _commit_noop)
    monkeypatch.setattr(te, "_emit_tool_complete_and_risk", _noop)
    monkeypatch.setattr(te, "_tool_progress_enabled", lambda agent: False)
    monkeypatch.setattr(te, "_budget_for_agent", lambda agent: None)
    return monkeypatch


def test_concurrent_interrupt_stamps_only_unstarted_calls(_env):
    """A mid-batch stop stamps an S3 resume checkpoint naming EXACTLY the un-started calls,
    never the one that completed (the #1695 discriminator)."""
    read = _Parsed("read_file", {"path": "/x"})
    search = _Parsed("search_files", {"pattern": "*.py"})
    write_c = _Parsed("write_file", {"path": "/y"})
    calls = [read, search, write_c]

    agent = _Agent()

    batch = types.SimpleNamespace(
        parsed_calls=calls,
        results=[_Done(read), None, None],       # only read_file started+completed
        timed_out_indices=set(),
        timeout_s=None,
    )
    te._append_batch_results(agent, [], "tid", batch, None)

    ck = getattr(agent, "_s3_pending_resume_checkpoint", None)
    assert ck is not None, "concurrent mid-batch interrupt must stamp a resume checkpoint"
    names = [t["name"] for t in ck["tool_calls"]]
    assert names == ["search_files", "write_file"], f"must stamp only un-started calls, got {names}"
    assert ck["kind"] == "tool_round"
    assert ck["round_start"] == 2  # 1-based index of first un-started call
    assert "read_file" not in names, "a completed call must never be resumable (re-run)"


def test_concurrent_interrupt_excludes_timed_out_slot(_env):
    """A timed-out slot (not a user interrupt) is NOT a resume candidate."""
    a = _Parsed("read_file", {"path": "/a"})
    b = _Parsed("search_files", {"pattern": "*.py"})
    agent = _Agent()
    batch = types.SimpleNamespace(
        parsed_calls=[a, b],
        results=[_Done(a), None],
        timed_out_indices={1},   # slot b timed out, not interrupted
        timeout_s=5.0,
    )
    te._append_batch_results(agent, [], "tid", batch, None)
    assert getattr(agent, "_s3_pending_resume_checkpoint", None) is None, \
        "timeout must not produce a resume checkpoint"


def test_concurrent_all_completed_no_stamp(_env):
    """If every call completed, nothing is resumable -> no stamp (node-boundary invariant)."""
    a = _Parsed("read_file", {"path": "/a"})
    b = _Parsed("write_file", {"path": "/b"})
    agent = _Agent()
    agent._interrupt_requested = False
    batch = types.SimpleNamespace(
        parsed_calls=[a, b],
        results=[_Done(a), _Done(b)],
        timed_out_indices=set(),
        timeout_s=None,
    )
    te._append_batch_results(agent, [], "tid", batch, None)
    assert getattr(agent, "_s3_pending_resume_checkpoint", None) is None
