"""S3 resume-prelude builder (agent.resume_prelude).

Continue=Resume re-runs the SAME interrupted tool call(s) instead of re-asking the model. The
turn loop already has the exact re-run seam: a ``prelude`` plays tool calls through the unchanged
``run_tool_round`` before the first model call and feeds real results back. This tests the pure gate
that maps a persisted resume-checkpoint to that prelude.
"""
from agent.resume_prelude import (
    build_resume_prelude,
    has_resumable_checkpoint,
    pending_tool_calls,
)

CHECKPOINT = {
    "kind": "tool_round",
    "round_start": 2,
    "tool_calls": [
        {"name": "search_files", "arguments": {"pattern": "*.py"}},
        {"name": "bash", "arguments": "ls -la"},
    ],
}


def test_has_resumable_checkpoint():
    assert has_resumable_checkpoint(CHECKPOINT) is True
    assert has_resumable_checkpoint(None) is False
    assert has_resumable_checkpoint({}) is False
    assert has_resumable_checkpoint({"tool_calls": []}) is False
    assert has_resumable_checkpoint("nonsense") is False


def test_pending_tool_calls_shape():
    assert pending_tool_calls(CHECKPOINT) == [
        {"name": "search_files", "arguments": {"pattern": "*.py"}},
        {"name": "bash", "arguments": "ls -la"},
    ]
    # no checkpoint / empty / no pending calls -> None (not resumable)
    assert pending_tool_calls(None) is None
    assert pending_tool_calls({}) is None
    assert pending_tool_calls({"tool_calls": []}) is None
    assert pending_tool_calls("nonsense") is None


def test_build_prelude_reruns_same_calls_in_order():
    prelude = build_resume_prelude(CHECKPOINT)
    assert prelude is not None
    steps = list(prelude)
    assert len(steps) == 2
    assert steps[0] == ("([resuming the interrupted tool round])", "search_files", {"pattern": "*.py"})
    assert steps[1] == ("([resuming the interrupted tool round])", "bash", "ls -la")


def test_build_prelude_none_for_nothing_to_resume():
    assert build_resume_prelude(None) is None
    assert build_resume_prelude({}) is None
    assert build_resume_prelude({"kind": "tool_round"}) is None  # no pending calls
    assert build_resume_prelude({"tool_calls": []}) is None


def test_build_prelude_skips_malformed_and_skins_args():
    prelude = build_resume_prelude({
        "tool_calls": [
            {"name": "", "arguments": {}},
            "not-a-dict",
            {"name": "read_file", "arguments": {"path": "/tmp/x.py"}},
            {"name": "bash", "arguments": "pwd"},  # args may be a string (as persisted)
        ],
    })
    assert prelude is not None
    steps = list(prelude)
    assert steps == [
        ("([resuming the interrupted tool round])", "read_file", {"path": "/tmp/x.py"}),
        ("([resuming the interrupted tool round])", "bash", "pwd"),
    ]
