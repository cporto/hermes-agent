"""Auth / rate-limit / upload+download coverage for the carried /v1/files* routes.

These routes are our own feature (upstream has no file-transfer API). The shared gate
``_files_route_prelude`` is a plain *sync* helper; a stray ``@_require_auth`` on it once made
every files route return HTTP 500 (fixed in efe9bec63f), so a structural regression test pins that.
All filesystem setup is hermetic: HOME and the allowlist roots point at a tmp dir.
"""

import inspect
import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import files_allowlist
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _FILES_UPLOAD_MAX_BYTES,
)

API_KEY = "-".join(("fixture", "files", "routes", "key", "123"))


def _adapter(*, key=API_KEY):
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))


def _app(adapter, *, client_max_size=None):
    kwargs = {} if client_max_size is None else {"client_max_size": client_max_size}
    app = web.Application(**kwargs)
    app.router.add_get("/v1/files", adapter._handle_files_download)
    app.router.add_post("/v1/files", adapter._handle_files_upload)
    app.router.add_post("/v1/files/validate", adapter._handle_files_validate)
    return app


def _auth(key=API_KEY):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def hermetic(monkeypatch, tmp_path):
    root = os.path.realpath(str(tmp_path))
    monkeypatch.setattr(files_allowlist, "_files_roots", lambda: [root])
    monkeypatch.setattr(files_allowlist, "_files_deny_dirs", list)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "good.txt").write_text("hello\n")
    return {"root": root, "good": str(tmp_path / "good.txt")}


# ----------------------------------------------------------------------
# Structural regression: the prelude must stay a plain sync helper
# ----------------------------------------------------------------------


def test_files_route_prelude_is_plain_sync_method():
    fn = APIServerAdapter._files_route_prelude
    assert not inspect.iscoroutinefunction(fn), "prelude must be sync (callers do not await it)"
    assert not hasattr(fn, "__wrapped__"), "prelude must not be wrapped by a decorator like @_require_auth"


def test_files_route_prelude_returns_tuple_not_coroutine():
    adapter = _adapter(key=None)  # no API key -> 403 branch; no request access needed
    out = adapter._files_route_prelude(object(), "files")
    assert isinstance(out, tuple) and len(out) == 2
    ctx, err = out
    assert ctx is None and err is not None and err.status == 403


# ----------------------------------------------------------------------
# Auth ladder: 403 no key -> 401 bad bearer
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_no_api_key_configured_is_403(hermetic):
    async with TestClient(TestServer(_app(_adapter(key=None)))) as client:
        resp = await client.get("/v1/files", params={"path": hermetic["good"]})
    assert resp.status == 403


@pytest.mark.asyncio
async def test_upload_no_api_key_configured_is_403(hermetic):
    async with TestClient(TestServer(_app(_adapter(key=None)))) as client:
        resp = await client.post("/v1/files", data=b"abc")
    assert resp.status == 403


@pytest.mark.asyncio
async def test_validate_no_api_key_configured_is_403(hermetic):
    async with TestClient(TestServer(_app(_adapter(key=None)))) as client:
        resp = await client.post("/v1/files/validate", json={"paths": [hermetic["good"]]})
    assert resp.status == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-key"}, {"Authorization": "Basic abc"}])
async def test_download_bad_or_missing_bearer_is_401(hermetic, headers):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.get("/v1/files", params={"path": hermetic["good"]}, headers=headers)
    assert resp.status == 401


@pytest.mark.asyncio
async def test_upload_bad_bearer_is_401(hermetic):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.post("/v1/files", data=b"abc", headers=_auth("wrong-key"))
    assert resp.status == 401


# ----------------------------------------------------------------------
# Download: allowlisted file vs outside the allowlist
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_allowlisted_file_200(hermetic):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.get("/v1/files", params={"path": hermetic["good"]}, headers=_auth())
        body = await resp.read()
    assert resp.status == 200
    assert body == b"hello\n"
    assert resp.headers["Content-Disposition"].startswith("attachment")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"


@pytest.mark.asyncio
async def test_download_outside_allowlist_is_403(hermetic):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.get("/v1/files", params={"path": "/etc/hosts"}, headers=_auth())
        text = await resp.text()
    assert resp.status == 403
    assert "/etc/hosts" not in text  # resolved path is never echoed


# ----------------------------------------------------------------------
# Rate limit: 30 requests / 60 s per action+profile, then 429 + Retry-After
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_31st_request_in_window_is_429_with_retry_after(hermetic):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        for i in range(30):
            resp = await client.get("/v1/files", params={"path": hermetic["good"]}, headers=_auth())
            assert resp.status == 200, f"request {i + 1} should pass"
            await resp.read()
        resp = await client.get("/v1/files", params={"path": hermetic["good"]}, headers=_auth())
    assert resp.status == 429
    assert resp.headers.get("Retry-After")


@pytest.mark.asyncio
async def test_rate_limit_is_per_action(hermetic):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        for _ in range(30):
            r = await client.get("/v1/files", params={"path": hermetic["good"]}, headers=_auth())
            await r.read()
        blocked = await client.get("/v1/files", params={"path": hermetic["good"]}, headers=_auth())
        up = await client.post("/v1/files", data=b"abc", headers=_auth())
    assert blocked.status == 429
    assert up.status == 201  # upload has its own bucket ("upload_f")


# ----------------------------------------------------------------------
# Upload: bounded body round-trips through GET; oversize rejected
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_then_download_roundtrip(hermetic):
    payload = b"\x00\x01binary-payload\n" * 8
    async with TestClient(TestServer(_app(_adapter()))) as client:
        up = await client.post(
            "/v1/files", data=payload,
            headers={**_auth(), "X-File-Filename": "My%20Report.txt", "Content-Type": "text/plain"})
        assert up.status == 201
        meta = await up.json()
        assert meta["size"] == len(payload)
        assert meta["name"].endswith(".txt")
        assert meta["path"].startswith(os.path.join(hermetic["root"], ".hermes", "uploads"))
        assert os.path.basename(meta["path"]).endswith(meta["name"])
        down = await client.get("/v1/files", params={"path": meta["path"]}, headers=_auth())
        got = await down.read()
    assert down.status == 200
    assert got == payload


@pytest.mark.asyncio
async def test_upload_active_extension_is_stored_as_bin(hermetic):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        up = await client.post("/v1/files", data=b"<script>1</script>",
                               headers={**_auth(), "X-File-Filename": "x.html"})
        meta = await up.json()
    assert up.status == 201
    assert meta["name"].endswith(".bin")


@pytest.mark.asyncio
async def test_upload_empty_body_is_400(hermetic):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.post("/v1/files", data=b"", headers=_auth())
    assert resp.status == 400


@pytest.mark.asyncio
async def test_upload_over_handler_cap_is_413(hermetic):
    # Raise the app's own body limit so the handler's explicit 10 MB check is what rejects it.
    big = b"x" * (_FILES_UPLOAD_MAX_BYTES + 1)
    async with TestClient(TestServer(_app(_adapter(), client_max_size=64 * 1024 * 1024))) as client:
        resp = await client.post("/v1/files", data=big, headers=_auth())
    assert resp.status == 413


@pytest.mark.asyncio
async def test_upload_over_app_body_limit_is_413(hermetic):
    # Default aiohttp client_max_size (1 MiB) trips HTTPRequestEntityTooLarge inside request.read().
    big = b"x" * (2 * 1024 * 1024)
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.post("/v1/files", data=big, headers=_auth())
    assert resp.status == 413
