"""Recovery of control-char-malformed JSON in coercion + delegate paths (regression)."""
from tools.arg_coercion import _coerce_json, _normalize_json_strings_for_schema
from tools.delegate_tool_tasks import _recover_tasks_from_json_string
from tools.json_escape import escape_raw_control_chars


def _raw(arr_with_newline):
    # Build a JSON array string that carries a REAL (unescaped) newline inside a string
    # value — the exact shape that made strict json.loads fail.
    return arr_with_newline


class TestJsonEscapeHelper:
    def test_escapes_literal_newline_inside_string(self):
        # real newline inside a string value
        s = '["a' + "\n" + 'b"]'
        assert escape_raw_control_chars(s) == '["a\\nb"]'

    def test_leaves_existing_escape_alone(self):
        s = '"a\\nb"'  # already-escaped newline (backslash + n)
        assert escape_raw_control_chars(s) == '"a\\nb"'


class TestCoercionRecovery:
    def test_coerce_json_recovers_literal_newline_array(self):
        raw = '["a.png", "b' + "\n" + 'c.png"]'
        assert _coerce_json(raw, list) == ["a.png", "b\nc.png"]

    def test_normalize_json_strings_recovers_literal_newline(self):
        schema = {"type": "array", "items": {"type": "string"}}
        out = _normalize_json_strings_for_schema('["git status", "bun' + "\n" + 'test"]', schema)
        assert out == ["git status", "bun\ntest"]

    def test_valid_escaped_json_not_double_escaped(self):
        assert _coerce_json('["a\\nb"]', list) == ["a\nb"]


class TestDelegateRecoverySharedHelper:
    def test_delegate_recovery_uses_shared_helper(self):
        raw = '[{"goal": "run the audit", "context": "line one' + "\n" + 'line two"}]'
        parsed, err = _recover_tasks_from_json_string(raw)
        assert err is None
        assert parsed == [{"goal": "run the audit", "context": "line one\nline two"}]
