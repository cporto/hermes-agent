"""Unit tests for the S1 stop-word classifier (agent/stopwords.py)."""
import pytest

from agent.stopwords import is_stop_word


@pytest.mark.parametrize("text", [
    "stop",
    "Stop.",
    "  quit  ",
    "please stop",
    "stop now",
    "cancel",
    "halt",
])
def test_is_stop_word_true(text):
    assert is_stop_word(text) is True


@pytest.mark.parametrize("text", [
    "stop when done",
    "don't stop",
    "continue working",
    "stopping",
    "stopped",
    "please stop doing that",
    "",
    None,
    "hello",
])
def test_is_stop_word_false(text):
    assert is_stop_word(text) is False
