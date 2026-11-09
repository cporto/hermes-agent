"""Behavioral tests for /v1/mercury/saved-prompts (GET/PUT) — Mercury's durable
saved-prompts store. Uses a temp file path (monkeypatched) so it never touches
real user data under the gateway home."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _make_adapter


def _build_app(adapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/mercury/saved-prompts", adapter._handle_saved_prompts_get)
    app.router.add_put("/v1/mercury/saved-prompts", adapter._handle_saved_prompts_put)
    return app


@pytest.mark.asyncio
async def test_saved_prompts_roundtrip(tmp_path):
    adapter = _make_adapter()
    adapter._saved_prompts_path = lambda: tmp_path / "sp.json"
    async with TestClient(TestServer(_build_app(adapter))) as cli:
        # Empty store -> empty list.
        r = await cli.get("/v1/mercury/saved-prompts")
        assert r.status == 200, await r.text()
        assert (await r.json())["prompts"] == []

        # PUT persists.
        r = await cli.put(
            "/v1/mercury/saved-prompts",
            json={"prompts": ["Summarize this thread", "draft follow-up"]},
        )
        assert r.status == 200, await r.text()
        body = await r.json()
        assert body["ok"] is True and body["count"] == 2

        # GET reflects the write and the file exists on disk.
        r = await cli.get("/v1/mercury/saved-prompts")
        assert (await r.json())["prompts"] == ["Summarize this thread", "draft follow-up"]
        assert (tmp_path / "sp.json").exists()


@pytest.mark.asyncio
async def test_saved_prompts_validation(tmp_path):
    adapter = _make_adapter()
    adapter._saved_prompts_path = lambda: tmp_path / "sp.json"
    async with TestClient(TestServer(_build_app(adapter))) as cli:
        # Non-string element -> 400.
        r = await cli.put("/v1/mercury/saved-prompts", json={"prompts": ["ok", 5]})
        assert r.status == 400, await r.text()
        # Missing body / wrong shape -> 400.
        r = await cli.put("/v1/mercury/saved-prompts", json={})
        assert r.status == 400, await r.text()
        r = await cli.put("/v1/mercury/saved-prompts", json={"prompts": "nope"})
        assert r.status == 400, await r.text()
        # Oversized single entry (over _SAVED_PROMPT_MAX_LEN) -> 400.
        r = await cli.put(
            "/v1/mercury/saved-prompts",
            json={"prompts": ["x" * (adapter._SAVED_PROMPT_MAX_LEN + 1)]},
        )
        assert r.status == 400, await r.text()
        # After failed writes the store is still empty (no partial write).
        r = await cli.get("/v1/mercury/saved-prompts")
        assert (await r.json())["prompts"] == []
        assert not (tmp_path / "sp.json").exists()
