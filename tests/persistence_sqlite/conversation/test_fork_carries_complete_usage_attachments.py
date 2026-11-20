"""Fork carries re-ported onto the persistence layer (Mercury depends on all three).

* ``sessions.complete``: the user-marked-done flag, lineage state like ``pinned`` / ``hidden``.
* ``sessions.context_usage``: the last context-gauge snapshot (JSON) so a client can draw it on load.
* ``messages.attachments``: image refs persisted beside the ``[screenshot]`` text projection.
"""

import json

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


def _session(backend, scope):
    store = backend.open(scope)
    store.sessions.create(NewSession("s1", "cli"))
    return store


def test_complete_flag_round_trips_and_is_on_the_session_row(backend, scope_a):
    store = _session(backend, scope_a)
    assert store.sessions.get("s1").complete is False
    assert store.sessions.set_complete("s1", True)
    record = store.sessions.get("s1")
    assert record.complete is True
    assert record.as_session_row()["complete"] == 1
    assert store.sessions.set_complete("s1", False)
    assert store.sessions.get("s1").complete is False


def test_complete_does_not_touch_title_or_other_flags(backend, scope_a):
    store = _session(backend, scope_a)
    store.sessions.set_title("s1", "Check system availability")
    store.sessions.set_pinned("s1", True)
    store.sessions.set_complete("s1", True)
    record = store.sessions.get("s1")
    assert record.title == "Check system availability"
    assert record.pinned is True and record.complete is True


def test_context_usage_snapshot_persists_as_json(backend, scope_a):
    store = _session(backend, scope_a)
    assert store.sessions.get("s1").context_usage is None
    payload = {"used": 1200, "window": 200000, "model": "m", "threshold": 150000, "updated_at": 1.5}
    assert store.sessions.update_context_usage("s1", payload)
    stored = store.sessions.get("s1").context_usage
    assert json.loads(stored) == payload
    assert json.loads(store.sessions.get("s1").as_session_row()["context_usage"]) == payload
    assert not store.sessions.update_context_usage("missing", payload)


def test_attachments_round_trip_through_transcript(backend, scope_a):
    store = _session(backend, scope_a)
    refs = [{"url": "data:image/png;base64,AAAA", "mime": "image/png"}]
    store.messages.append("s1", NewMessage(role="user", content="[screenshot]", attachments=refs))
    store.messages.append("s1", NewMessage(role="assistant", content="seen"))
    rows = store.messages.read_transcript("s1")
    assert [r.attachments for r in rows] == [refs, None]
    assert rows[0].as_wire_row()["attachments"] == refs
    assert rows[0].content == "[screenshot]"


def test_attachments_are_extracted_from_image_parts_when_not_given(backend, scope_a):
    store = _session(backend, scope_a)
    content = [{"type": "text", "text": "look"},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]
    store.messages.append("s1", NewMessage(role="user", content=content))
    (row,) = store.messages.read_transcript("s1")
    assert row.attachments == [{"url": "data:image/jpeg;base64,QUJD", "mime": "image/jpeg"}]


def test_attachments_survive_compaction_clone(backend, scope_a):
    store = _session(backend, scope_a)
    refs = [{"url": "data:image/png;base64,ZZ", "mime": "image/png"}]
    store.messages.append("s1", NewMessage(role="user", content="[screenshot]", attachments=refs))
    rows = store.messages.read_transcript("s1")
    replacement = [NewMessage.from_live(rows[0].as_wire_row())]
    store.messages.compact("s1", replacement)
    assert [r.attachments for r in store.messages.read_transcript("s1")] == [refs]
