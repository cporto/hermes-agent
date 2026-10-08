"""Behavioral test for the api_server interactive-clarify bridge.

Covers the NEW Mercury ClarifyCard surface: a pending clarify registered via
tools.clarify_gateway (the same primitive the native SSE's _clarify_callback_sync
uses) is resolved through the HTTP POST /api/sessions/{id}/clarify route, the
blocking waiter unblocks with the user's answer, and an already-resolved/unknown
id fails closed with 409.

This is a behavior contract (resolve -> waiter unblocks, fail-closed re-resolve),
NOT a change-detector and it never reads source in the test.
"""
import threading

import pytest
from aiohttp import ClientSession

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools import clarify_gateway as clarify_mod


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def _create_session(client, port, key) -> str:
    async with client.post(
        f"http://127.0.0.1:{port}/api/sessions",
        headers={**_auth(key), "Content-Type": "application/json"},
        json={},
    ) as response:
        assert response.status == 201, await response.text()
        data = await response.json()
        sess = data.get("session") or {}
        return str(sess.get("id") or data.get("id") or data.get("session_id"))


@pytest.mark.asyncio
async def test_clarify_resolve_route_unblocks_waiter_and_fails_closed():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "key": "test-clarify-key",
    }))
    assert await adapter.connect()
    session_id = None
    clarify_id = "clr_test_abc123"
    try:
        port = adapter._site._server.sockets[0].getsockname()[1]
        async with ClientSession() as client:
            session_id = await _create_session(client, port, "test-clarify-key")

            # Register a pending clarify exactly as _clarify_callback_sync does.
            clarify_mod.register(
                clarify_id=clarify_id, session_key=session_id,
                question="Which environment?", choices=["staging", "prod"],
            )

            # A waiter blocks on the threading.Event until the route resolves it.
            result_box: dict = {}
            waiter = threading.Thread(
                target=lambda: result_box.update(
                    result=clarify_mod.wait_for_response(clarify_id, timeout=10)))
            waiter.start()

            # Resolve through the real HTTP route.
            async with client.post(
                f"http://127.0.0.1:{port}/api/sessions/{session_id}/clarify",
                headers={**_auth("test-clarify-key"), "Content-Type": "application/json"},
                json={"clarify_id": clarify_id, "response": "prod"},
            ) as response:
                assert response.status == 200, await response.text()
                data = await response.json()
                assert data.get("resolved") is True

            waiter.join(timeout=15)
            assert not waiter.is_alive(), "waiter did not unblock after resolve"
            assert result_box.get("result") == "prod"

            # Fail-closed: the same id is already resolved -> 409, not re-delivery.
            async with client.post(
                f"http://127.0.0.1:{port}/api/sessions/{session_id}/clarify",
                headers={**_auth("test-clarify-key"), "Content-Type": "application/json"},
                json={"clarify_id": clarify_id, "response": "staging"},
            ) as response:
                assert response.status == 409

            # Fail-closed: validation rejects a missing/empty id.
            async with client.post(
                f"http://127.0.0.1:{port}/api/sessions/{session_id}/clarify",
                headers={**_auth("test-clarify-key"), "Content-Type": "application/json"},
                json={"response": "x"},
            ) as response:
                assert response.status == 400
    finally:
        clarify_mod.clear_session(session_id or "")
        if session_id:
            try:
                port = adapter._site._server.sockets[0].getsockname()[1]
                async with ClientSession() as client:
                    async with client.delete(
                        f"http://127.0.0.1:{port}/api/sessions/{session_id}",
                        headers=_auth("test-clarify-key"),
                    ) as response:
                        response.status  # best-effort cleanup
            except Exception:
                pass
        await adapter.disconnect()
        adapter._site = None  # allow a second connect in the same process if reused
