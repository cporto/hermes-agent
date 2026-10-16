"""R6 regression: portability import collapses duplicate steer rows by content+timestamp.

A donor that pre-dates the durable-uid fix (or a mass re-import of the same session) can carry
multiple active display_kind='steer' rows for one logical steer. Imports don't carry message_uid
(added fresh at INSERT), so those duplicates would import as distinct active steers and bypass the
write-side dedupe/unique-index collapse. _collate_duplicate_import_steers must collapse them.
"""
import pytest


def _msg(content, ts, role="user", active=1, display_kind=None, **kw):
    m = {"role": role, "content": content, "timestamp": ts, "active": active}
    if display_kind is not None:
        m["display_kind"] = display_kind
    m.update(kw)
    return m


def test_collate_keeps_one_duplicate_steer():
    from hermes_state_portability import SessionPortabilityMixin
    rows = [
        _msg("  steer one  ", 100.0, display_kind="steer"),
        _msg("steer one", 100.0, display_kind="steer"),      # same content strip + ts -> dup
        _msg("steer two", 102.0, display_kind="steer"),
        _msg("normal user msg", 103.0, role="user"),          # passthrough untouched
        _msg("assistant reply", 104.0, role="assistant"),
    ]
    out = SessionPortabilityMixin._collate_duplicate_import_steers(rows)
    steers = [m for m in out if m.get("display_kind") == "steer"]
    # two logical steers, the trailing duplicate dropped
    assert len(steers) == 2, f"want 2 steers, got {len(steers)}"
    # non-steer rows fully preserved and in order
    assert [m["content"] for m in out if m.get("display_kind") != "steer"] == [
        "normal user msg", "assistant reply",
    ]


def test_collate_distinct_steers_with_same_content_different_ts_kept():
    from hermes_state_portability import SessionPortabilityMixin
    rows = [
        _msg("steer one", 100.0, display_kind="steer"),
        _msg("steer one", 200.0, display_kind="steer"),      # same content, different ts -> NOT a dup
    ]
    out = SessionPortabilityMixin._collate_duplicate_import_steers(rows)
    assert len([m for m in out if m.get("display_kind") == "steer"]) == 2


def test_collate_no_op_when_no_steers():
    from hermes_state_portability import SessionPortabilityMixin
    rows = [_msg("a", 1.0), _msg("b", 2.0, role="assistant"), _msg("c", 3.0, role="tool")]
    assert SessionPortabilityMixin._collate_duplicate_import_steers(rows) == rows


def test_normalize_import_session_collapses_steers():
    from hermes_state_portability import SessionPortabilityMixin
    importer = SessionPortabilityMixin()
    raw = {
        "id": "s1", "messages": [
            _msg("[OUT-OF-BAND]\nsteer one\n[/OUT-OF-BAND]", 100.0, display_kind="steer"),
            _msg("[OUT-OF-BAND]\nsteer one\n[/OUT-OF-BAND]", 100.0, display_kind="steer"),
        ],
    }
    item = importer._normalize_import_session(raw, "s1", raw["messages"])
    steers = [m for m in item["messages"] if m.get("display_kind") == "steer"]
    assert len(steers) == 1
