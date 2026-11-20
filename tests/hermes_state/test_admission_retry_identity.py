"""Idempotent retry identity must ignore the per-request ``approval_session_key``.

``api_turn_v1.approval_session_key`` is a fresh run id minted for every HTTP request. It is stored with
the admission (the producer binds it) but must not count toward request identity: a retry carrying the
same Idempotency-Key replays the first admission, while a genuinely different request still conflicts.
"""
import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB


def _payload(text, key):
    return {'text': text, 'api_turn_v1': {'history': None, 'settings': {}, **({'approval_session_key': key} if key else {})}}


def test_request_identity_payload_drops_only_the_per_request_key():
    from hermes_state_keys import request_identity_payload
    p = _payload('hi', 'chatcmpl-aaa')
    assert request_identity_payload(p) == _payload('hi', None)
    assert p['api_turn_v1']['approval_session_key'] == 'chatcmpl-aaa', 'must not mutate the stored payload'
    for plain in ({'text': 'x'}, {'api_turn_v1': 'oops'}, None, 'str'):
        assert request_identity_payload(plain) == plain


def test_exact_retry_with_a_new_approval_session_key_replays_the_first_admission(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('s', source='cli')
        first = rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='s', request_id='idem-1',
                                       payload=_payload('do the thing', 'chatcmpl-first'))
        retry = rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='s', request_id='idem-1',
                                       payload=_payload('do the thing', 'chatcmpl-second'))
        assert retry['admission_id'] == first['admission_id']


def test_same_idempotency_key_with_a_different_message_still_conflicts(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('s', source='cli')
        rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='s', request_id='idem-1',
                               payload=_payload('do the thing', 'chatcmpl-first'))
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='s', request_id='idem-1',
                                   payload=_payload('do a DIFFERENT thing', 'chatcmpl-first'))


def test_stored_payload_still_carries_the_key_the_producer_binds(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('s', source='cli')
        row = rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='s', request_id='idem-1',
                                     payload=_payload('hi', 'chatcmpl-first'))
        stored = rt.get_session_admission(db, admission_id=row['admission_id'])
        assert 'chatcmpl-first' in str(stored)
