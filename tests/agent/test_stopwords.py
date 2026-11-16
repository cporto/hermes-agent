"""Unit tests for the S1 stop-word + S3 continue-word classifier (agent/stopwords.py)."""
import pytest

from agent.stopwords import is_continue_word, is_stop_word


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


@pytest.mark.parametrize("text", [
    "continue",
    "Continue.",
    "  resume  ",
    "go on",
    "keep going",
    "carry on",
    "continue!",
    "cOntInUe",
])
def test_is_continue_word_true(text):
    assert is_continue_word(text) is True


@pytest.mark.parametrize("text", [
    "continue working",
    "please continue",
    "continue the report",
    "keep going please",
    "carry on please",
    "resume the task",
    "",
    None,
    "stop",
    "halt",
    "hello",
])
def test_is_continue_word_false(text):
    assert is_continue_word(text) is False
