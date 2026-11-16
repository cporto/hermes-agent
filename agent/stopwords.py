"""Deterministic stop/continue-word classifier for the S1 cancel + S3 resume control planes.

This is the "runtime check, not a prompt ask": a user message that classifies as a stop word
becomes a runtime Cancel before it ever reaches the model; a message that classifies as a
continue word after an interrupted turn becomes a RESUME, rehydrating the previously persisted
checkpoint instead of a fresh follow-up turn. Pure, deterministic, business-safe, no LLM, no NLU.
False-negative (no match) degrades to today's behaviour (safe).

#S1: a natural-language composer message ("stop", "halt", "cancel") is enforced as a REAL runtime
Cancel OUTSIDE the model — it aborts the in-flight LLM+tool loop, prevents queued steps, is checked
in the runtime BEFORE the model prompt is forwarded, and fails safe to "stopped". "Stop" is never
forwarded to the model as a normal prompt.

#S3: a bare "continue"/"resume" after a stop/cancel rehydrates the persisted checkpointer snapshot
of the interrupted run (the Continue=Resume half of the Stop/Continue pair). This module is the
single source of truth for BOTH classifiers: the same allowlist discipline bounds false-positives.
"""

from __future__ import annotations


_STOP_WORDS: frozenset[str] = frozenset({
    "stop", "stop.", "halt", "cancel", "quit", "stop now", "please stop",
})

# S3 continuation allowlist (mirrors _STOP_WORDS): a bare continue/resume control token. Exact +
# punctuation-only, so "continue the task" / "please continue" do NOT match — only a message that
# is exactly a control token with optional surrounding whitespace/punctuation. A false-negative
# degrades to a normal follow-up turn (safe); the narrow allowlist bounds false-positives the same
# way the stop allowlist does.
_CONTINUE_WORDS: frozenset[str] = frozenset({
    "continue", "resume", "go on", "keep going", "carry on", "continue.", "resume.",
})


def is_stop_word(text: str) -> bool:
    """True when *text* is exactly a stop command (trimmed, lowercased, whitespace/punct-normalised).

    Exact + punctuation-only: "stop when done" / "don't stop" / "continue" are NOT stops. Only a
    message that is X (a stop token) with optional surrounding whitespace/punctuation matches.
    """
    return _token_in(text, _STOP_WORDS)


def is_continue_word(text: str) -> bool:
    """True when *text* is exactly a continue/resume command (trimmed, lowercased,
    whitespace/punct-normalised).

    Exact + punctuation-only: "continue working" / "please continue" / "continue the report" are
    NOT continues. Only a bare control token ("continue", "resume", "go on", "keep going",
    "carry on") with optional surrounding whitespace/punctuation matches — the S3 Continue=Resume
    control token. A false-negative degrades to today's normal follow-up turn.
    """
    return _token_in(text, _CONTINUE_WORDS)


def _token_in(text: str, words: frozenset[str]) -> bool:
    """Shared exact-token membership: trim, lowercase, normalise em-dash, drop surrounding
    punctuation-only characters, then compare."""

    if not isinstance(text, str):
        return False
    token = text.strip().lower()
    token = token.replace("\u2014", "-").strip()
    token = token.strip(" .,!?;:-\u2014")
    return token in words
