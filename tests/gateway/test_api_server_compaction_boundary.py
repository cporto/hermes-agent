"""GAP-A: a standalone compaction carrier is flagged as a within-session boundary;
a merged carrier (which keeps real prior-tail content) is NOT flagged."""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


STANDALONE_SUMMARY = (
    f"{SUMMARY_PREFIX}\n\n"
    f"{HISTORICAL_TASK_HEADING}\nold work\n\n"
    f"{_SUMMARY_END_MARKER}"
)
MERGED_CARRIER = (
    f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"
    "Refactor complete.\n\n"
    f"{_MERGED_SUMMARY_DELIMITER}\n\n"
    f"{STANDALONE_SUMMARY}"
)


def _row(role: str, content, **extra) -> dict:
    row = {"id": 1, "session_id": "s1", "role": role, "content": content}
    row.update(extra)
    return row


@pytest.fixture(scope="module")
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def test_standalone_carrier_is_flagged_as_boundary(adapter):
    projected = APIServerAdapter._message_response(
        _row("user", STANDALONE_SUMMARY, tool_calls=[{"id": "stale"}])
    )
    assert projected["display_kind"] == "hidden"
    assert projected["content"] == ""
    assert projected["compaction_boundary"] is True


def test_merged_carrier_is_not_flagged(adapter):
    projected = APIServerAdapter._message_response(
        _row("assistant", MERGED_CARRIER, tool_calls=[{"id": "prior-call"}])
    )
    assert projected["content"] == "Refactor complete."
    assert "compaction_boundary" not in projected
