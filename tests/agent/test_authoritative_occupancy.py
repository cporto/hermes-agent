"""Tests for the authoritative context-occupancy source used by the gauge persist.

The Mercury header "context ring" gauge must reflect CURRENT occupancy from the
authoritative compressor (``last_prompt_tokens`` clamped to the window), never the
cumulative billing sum (``session_prompt_tokens`` / ``input_tokens``): a running +=
exceeds the window on long sessions (false >100%), and a frozen used=0 read as 0%.
"""

from unittest.mock import MagicMock

from agent.context_breakdown import get_authoritative_occupancy

def test_returns_compressor_last_prompt_tokens_clamped_to_window():
    """(a) Occupancy comes from the compressor, clamped to the context window."""
    agent = MagicMock()
    agent.context_compressor = MagicMock(
        context_length=262_144, last_prompt_tokens=400_000,
        last_real_prompt_tokens=150_000)
    agent._last_turn_usage = {"prompt_tokens": 999_999}

    fields = get_authoritative_occupancy(agent)

    # The cumulative-looking 400k / 999k figures never read over the window.
    assert fields["context_used"] == 262_144
    assert fields["context_max"] == 262_144
    assert fields["context_percent"] == 100

def test_returns_empty_when_no_compressor_and_no_last_turn_usage():
    """(b) Unknown occupancy omits ({}), never reports 0."""
    agent = MagicMock()
    agent.context_compressor = None
    agent._last_turn_usage = None

    fields = get_authoritative_occupancy(agent)

    assert fields == {}

def test_fallback_last_turn_usage_clamps_to_window():
    """(c) Clamps when the fallback last-turn prompt_tokens exceeds the window."""
    agent = MagicMock()
    agent.context_compressor = None
    agent.context_length = 262_144
    agent._last_turn_usage = {"prompt_tokens": 500_000}

    fields = get_authoritative_occupancy(agent)

    assert fields["context_used"] == 262_144
    assert fields["context_max"] == 262_144
    assert fields["context_percent"] == 100

def test_unknown_compressor_but_empty_last_turn_usage_omits():
    """A compressor that measures nothing must not leak a bogus number."""
    agent = MagicMock()
    agent.context_compressor = MagicMock(context_length=0, last_prompt_tokens=0)
    agent._last_turn_usage = {}

    assert get_authoritative_occupancy(agent) == {}
