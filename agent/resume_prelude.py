"""S3 Continue=Resume: build the scripted prelude that re-runs an interrupted tool round.

Continuation model (industry-settled): LangGraph checkpointer + CopilotKit useInterrupt persist the
pending tool call at interrupt; "continue" rehydrates and re-RUNS that same call rather than asking
the model to start fresh. Hermes already has the exact re-run seam: a ``prelude`` is a generator of
``(content, tool_name, args)`` that the turn loop plays through the unchanged ``run_tool_round``
BEFORE the first model call (agent/turn_scripted_prelude.py), appends the real tool results, then
returns control to the model. So a resumed turn is a normal user turn whose ``prelude`` is the
checkpoint's pending tool calls.

This module maps a persisted resume-checkpoint (SessionStore.set_resume_checkpoint output) to that
prelude. Pure and deterministic (no I/O), so it is unit-testable without a live agent.
"""
from __future__ import annotations

from typing import Any, Generator, Optional

from agent.turn_scripted_prelude import Prelude

# Assistant-facing interim text for each re-run step. Kept terse and neutral; never carries raw
# chain-of-thought (invariant: CoT never enters replayable content, #81841).
_REPLAY_INTERIM = "([resuming the interrupted tool round])"


def pending_tool_calls(checkpoint: Optional[dict[str, Any]]) -> Optional[list]:
    """The checkpoint's pending tool calls as ``[{name, arguments}, ...]``, or ``None`` when there
    is no resumable checkpoint at all (absent, empty, wrong type, or no pending calls)."""
    if not isinstance(checkpoint, dict):
        return None
    calls = checkpoint.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    return calls


def has_resumable_checkpoint(checkpoint: Optional[dict[str, Any]]) -> bool:
    """True when *checkpoint* carries at least one pending tool call to re-run."""
    return bool(pending_tool_calls(checkpoint))


def build_resume_prelude(
    checkpoint: Optional[dict[str, Any]], *, interim: str = _REPLAY_INTERIM,
) -> Optional[Prelude]:
    """Generator prelude that plays the checkpoint's pending tool calls in order.

    Each step yields ``(interim, tool_name, args)``; the loop feeds each tool's real result back and
    continues to the next call, then returns control to the model exactly as a normal tool round
    does. Returns ``None`` when there is nothing resumable (so a stale/absent checkpoint degrades to
    a plain follow-up turn — no false "resumed")."""
    calls = pending_tool_calls(checkpoint)
    if not calls:
        return None

    # Validate/skin each entry deterministically; skip malformed ones. Args are passed straight to
    # the tool (same json round-trip the normal round applies).
    skinned: list[tuple[str, Any]] = []
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name")
        if isinstance(name, str) and name:
            skinned.append((name, tc.get("arguments")))

    def _prelude() -> Generator[tuple[str, str, dict], Optional[str], None]:
        for name, args in skinned:
            yield interim, name, dict(args) if isinstance(args, dict) else args

    return _prelude() if skinned else None
