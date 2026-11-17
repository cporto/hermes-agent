"""S3 resume-checkpoint stamp — shared interrupt funnel (the concurrent/segmented path fix).

Root cause found live (Fable Gate 4): the resume checkpoint was only stamped in
`_execute_tool_calls_sequential`. Real multi-tool rounds dispatch to the concurrent/segmented
executor, whose interrupt path calls `_append_skipped_tool_results` directly — so no checkpoint
was ever written and "continue" degraded to a plain redo. This test pins the fix: the shared
funnel stamps the skipped (pending, un-executed) calls so ALL executor paths capture a resumable
checkpoint, and the one-shot guard preserves the sequential path's more precise subset.
"""
from agent.tool_executor import _append_skipped_tool_results


class _Fn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


class _TC:
    def __init__(self, name, args):
        self.function = _Fn(name, args)


class _Agent:
    def __init__(self):
        self._s3_pending_resume_checkpoint = None
        self._vprint = lambda *a, **k: None
        self.log_prefix = ""
        self.time = 0.0


class _Messages:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


def _tc_name(tc):
    return tc.function.name


def test_concurrent_interrupt_stamps_resume_checkpoint():
    agent = _Agent()
    calls = [_TC("search_files", {"p": "*.py"}), _TC("read_file", {"path": "/x"})]
    # The concurrent executor calls _append_skipped_tool_results DIRECTLY (no prior stamp).
    # It is the real funnel after the tool executor has unwound past parse/etc.
    ok = _append_skipped_tool_results(
        agent, _Messages(), calls, "tid",
        content="[skipped {name}]", hook_error_type="user_interrupt", flush_stage=None,
    )
    assert ok is True
    ck = getattr(agent, "_s3_pending_resume_checkpoint", None)
    assert ck is not None, "concurrent-interrupt path must stamp a resume checkpoint"
    assert ck["kind"] == "tool_round"
    assert ck["tool_calls"] == [
        {"name": "search_files", "arguments": {"p": "*.py"}},
        {"name": "read_file", "arguments": {"path": "/x"}},
    ]


def test_one_shot_guard_preserves_existing_stamp():
    agent = _Agent()
    # A prior (sequential) interrupt already stamped the pending subset.
    agent._s3_pending_resume_checkpoint = {
        "kind": "tool_round", "round_start": 2,
        "tool_calls": [{"name": "read_file", "arguments": {"path": "/x"}}],
    }
    calls = [_TC("search_files", {"p": "*.py"}), _TC("read_file", {"path": "/x"})]
    _append_skipped_tool_results(
        agent, _Messages(), calls, "tid",
        content="[skipped {name}]", hook_error_type="user_interrupt", flush_stage=None,
    )
    # The one-shot guard must NOT overwrite the more-precise sequential subset.
    assert getattr(agent, "_s3_pending_resume_checkpoint", None)["tool_calls"] == [
        {"name": "read_file", "arguments": {"path": "/x"}},
    ]
