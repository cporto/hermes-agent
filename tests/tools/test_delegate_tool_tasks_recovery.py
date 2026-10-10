from tools.delegate_tool_tasks import _recover_tasks_from_json_string


def test_recovery_escapes_literal_newline_in_string_value():
    # The exact "Expecting ',' delimiter" case: a real (unescaped) newline
    # inside a string value of the string-shaped tasks payload.
    raw = '[{"goal": "run the audit", "context": "line one\nline two"}]'
    parsed, err = _recover_tasks_from_json_string(raw)
    assert err is None
    assert parsed == [{"goal": "run the audit", "context": "line one\nline two"}]


def test_strictly_valid_json_string_still_works():
    # A proper \n escape must NOT be double-escaped.
    raw = '[{"goal": "do a thing", "context": "a\\nb"}]'
    parsed, err = _recover_tasks_from_json_string(raw)
    assert err is None
    assert parsed == [{"goal": "do a thing", "context": "a\nb"}]


def test_genuinely_invalid_string_returns_error():
    parsed, err = _recover_tasks_from_json_string("this is not json")
    assert parsed is None
    assert err is not None
    assert "could not be parsed" in err


def test_non_string_tasks_passthrough():
    parsed, err = _recover_tasks_from_json_string([{"goal": "g"}])
    assert parsed is None
    assert err is None
