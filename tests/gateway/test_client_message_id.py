"""Regression tests for the client_message_id idempotency key (exact optimistic-message match).

Mercury mints a per-send key, the gateway persists it on the durable user row's
display_metadata, and the client matches its optimistic turn by EXACT id. These tests pin the
two security-relevant edges: the door validator and the whitelisted surface (no metadata leak).
"""
from gateway.platforms.api_server import APIServerAdapter, _valid_client_message_id


def test_valid_client_message_id_accepts_safe_tokens():
    assert _valid_client_message_id("550e8400-e29b-41d4-a716-446655440000") == "550e8400-e29b-41d4-a716-446655440000"
    assert _valid_client_message_id("ck-abc_123-XYZ") == "ck-abc_123-XYZ"
    assert _valid_client_message_id("a") == "a"


def test_valid_client_message_id_rejects_unsafe_values():
    for bad in ["has spaces", "a/b", "a" * 65, "", "drop;table", "a\nb", 123, None, ["x"], {"a": 1}]:
        assert _valid_client_message_id(bad) is None


def test_message_response_surfaces_client_message_id_only():
    row = {
        "id": 7, "session_id": "s", "role": "user", "content": "hi",
        "display_metadata": {
            "client_message_id": "ck-1",
            "gateway_input_owner": "principal:secret",
            "notification_category": "diagnostic",
            "delegation_id": "d-9",
        },
    }
    resp = APIServerAdapter._message_response(row)
    assert resp["client_message_id"] == "ck-1"
    # The whole display_metadata blob must NOT leak; only the whitelisted scalar surfaces.
    assert "display_metadata" not in resp
    assert "gateway_input_owner" not in resp
    assert "notification_category" not in resp
    assert "delegation_id" not in resp


def test_message_response_omits_key_when_absent():
    row = {"id": 8, "session_id": "s", "role": "user", "content": "hi"}
    resp = APIServerAdapter._message_response(row)
    assert "client_message_id" not in resp


def test_message_response_omits_key_when_empty():
    row = {"id": 9, "session_id": "s", "role": "user", "content": "hi",
           "display_metadata": {"client_message_id": ""}}
    resp = APIServerAdapter._message_response(row)
    assert "client_message_id" not in resp
