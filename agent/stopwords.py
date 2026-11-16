"""Deterministic stop-word classifier for the S1 runtime cancel control plane.

This is the "runtime check, not a prompt ask": a user message that classifies as a stop word
becomes a runtime Cancel before it ever reaches the model. #S1: a natural-language composer
message ("stop", "halt", "cancel") is enforced as a REAL runtime Cancel OUTSIDE the model — it
aborts the in-flight LLM+tool loop, prevents queued steps, is checked in the runtime BEFORE the
model prompt is forwarded, and fails safe to "stopped". "Stop" is never forwarded to the model as
a normal prompt. This module is the single source of truth for that classification: pure,
deterministic, business-safe, no LLM, no NLU. False-negative (no match) degrades to today's
behaviour (safe); the narrow allowlist + requirement of an active run bounds false-positives.
"""

from __future__ import annotations


_STOP_WORDS: frozenset[str] = frozenset({
    "stop", "stop.", "halt", "cancel", "quit", "stop now", "please stop",
})


def is_stop_word(text: str) -> bool:
    """True when *text* is exactly a stop command (trimmed, lowercased, whitespace/punct-normalised).

    Exact + punctuation-only: "stop when done" / "don't stop" / "continue" are NOT stops. Only a
    message that is X (a stop token) with optional surrounding whitespace/punctuation matches.
    """
    if not isinstance(text, str):
        return False
    token = text.strip().lower()
    token = token.replace("\u2014", "-").strip()
    # punctuation-only trailing/leading allowed (stop., Stop!,  quit )
    token = token.strip(" .,!?;:\u2014-")
    return token in _STOP_WORDS
