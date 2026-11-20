"""Idempotent retry identity must ignore the per-request ``approval_session_key``.

``api_turn_v1.approval_session_key`` is a fresh run id minted for every HTTP request. It is stored with
the admission (the producer binds it) but must not count toward request identity: a retry carrying the
same Idempotency-Key replays the first admission, while a genuinely different request still conflicts.
"""

from __future__ import annotations

import pytest

from persistence.authority.vocabulary import request_identity_payload
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.authority._admission_support import EPOCH


def _payload(text, key):
    return {"text": text, "api_turn_v1": {"history": None, "settings": {}, **({"approval_session_key": key} if key else {})}}


def _admit(store, text, key, request_id="idem-1"):
    record, created = store.admissions.admit(epoch=EPOCH, principal_id="api", session_id="s", request_id=request_id,
                                             payload=_payload(text, key))
    return record, created


def test_request_identity_payload_drops_only_the_per_request_key():
    p = _payload("hi", "chatcmpl-aaa")
    assert request_identity_payload(p) == _payload("hi", None)
    assert p["api_turn_v1"]["approval_session_key"] == "chatcmpl-aaa", "must not mutate the stored payload"
    for plain in ({"text": "x"}, {"api_turn_v1": "oops"}, None, "str"):
        assert request_identity_payload(plain) == plain


def test_exact_retry_with_a_new_approval_session_key_replays_the_first_admission(seeded):
    _, store, _ = seeded(lambda db, epoch: db.create_session("s", source="cli"))
    first, created = _admit(store, "do the thing", "chatcmpl-first")
    retry, retry_created = _admit(store, "do the thing", "chatcmpl-second")
    assert created and not retry_created
    assert retry.admission_id == first.admission_id


def test_same_idempotency_key_with_a_different_message_still_conflicts(seeded):
    _, store, _ = seeded(lambda db, epoch: db.create_session("s", source="cli"))
    _admit(store, "do the thing", "chatcmpl-first")
    with pytest.raises(AuthorityRefusal) as refused:
        _admit(store, "do a DIFFERENT thing", "chatcmpl-first")
    assert refused.value.reason == "admission_conflict"
