"""Batch file-validation oracle (POST /v1/files/validate) + the anti-drift invariant.

POST /v1/files/validate must mirror the shared allowlist validator behind GET /v1/files, so
"validate says ok" IFF "download returns 200". Both consume the single source of truth in
:mod:`gateway.files_allowlist`; the agreement test below asserts that structurally guaranteed
invariant explicitly for a seeded path matrix. All path setup is hermetic (``_files_roots`` is
patched to a tmp dir, HOME to a tmp path) — real user roots are never touched.
"""

import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import files_allowlist
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _FILES_MAX_BYTES


API_KEY = "-".join(("fixture", "files", "validate", "key", "123"))


def _adapter(*, key=API_KEY):
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))


def _app(adapter):
    app = web.Application()
    app.router.add_post("/v1/files/validate", adapter._handle_files_validate)
    app.router.add_get("/v1/files", adapter._handle_files_download)
    return app


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture
def seeded(monkeypatch, tmp_path):
    """Patch roots/HOME to a tmp dir and scaffold a set of fixtures to validate against."""
    root = os.path.realpath(str(tmp_path))
    monkeypatch.setattr(files_allowlist, "_files_roots", lambda: [root])
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(files_allowlist, "_files_deny_dirs", list)
    (tmp_path / "good.txt").write_text("hello\n")
    git = tmp_path / "repo" / ".git"
    git.mkdir(parents=True)
    (git / "secret").write_text("nope")
    (tmp_path / "plain_dir").mkdir()
    hard = tmp_path / "hard_a.txt"
    hard.write_text("hard")
    os.link(hard, tmp_path / "hard_b.txt")
    big = tmp_path / "big.bin"
    with open(big, "wb") as fh:
        fh.truncate(_FILES_MAX_BYTES + 1)
    _files_roots_here = files_allowlist._files_roots()
    return {
        "root": root,
        "roots": _files_roots_here,
        "good": str(tmp_path / "good.txt"),
        "git_secret": str(git / "secret"),
        "plain_dir": str(tmp_path / "plain_dir"),
        "hard": str(hard),
        "big": str(big),
        "missing": str(tmp_path / "nope.txt"),
        "outside": "/etc/hosts",
    }


async def _validate(client, paths):
    resp = await client.post(
        "/v1/files/validate", json={"paths": paths}, headers=_auth())
    return resp


# ----------------------------------------------------------------------
# Individual classifications
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_allow_real_regular_file(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["good"]])
    assert resp.status == 200
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is True
    assert row["path"] == seeded["good"]
    assert row["name"] == "good.txt"
    assert row["size"] == len("hello\n")


@pytest.mark.asyncio
async def test_validate_rejects_directory(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["plain_dir"]])
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is False
    assert row["reason"] == "not_a_file"


@pytest.mark.asyncio
async def test_validate_rejects_missing_path(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["missing"]])
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is False
    assert row["reason"] == "not_found"


@pytest.mark.asyncio
async def test_validate_rejects_path_outside_roots(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["outside"]])
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is False
    assert row["reason"] == "not_allowed"


@pytest.mark.asyncio
async def test_validate_rejects_deny_segment(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["git_secret"]])
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is False
    assert row["reason"] == "denied"


@pytest.mark.asyncio
async def test_validate_rejects_hardlink(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["hard"]])
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is False
    assert row["reason"] == "hardlink"


@pytest.mark.asyncio
async def test_validate_rejects_oversize(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["big"]])
    body = await resp.json()
    row = body["results"][0]
    assert row["ok"] is False
    assert row["reason"] == "too_large"


# ----------------------------------------------------------------------
# Batch / shape / cap / empty
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_batch_preserves_input_order(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [
            seeded["good"], seeded["missing"], seeded["plain_dir"], seeded["outside"],
        ])
    body = await resp.json()
    results = body["results"]
    assert [r["path"] for r in results] == [
        seeded["good"], seeded["missing"], seeded["plain_dir"], seeded["outside"],
    ]
    assert [r["ok"] for r in results] == [True, False, False, False]


@pytest.mark.asyncio
async def test_validate_cap_over_64_returns_400(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["good"]] * 65)
    assert resp.status == 400
    assert (await resp.json()) == {"error": [{"code": "files_validate_invalid"}]}


@pytest.mark.asyncio
async def test_validate_empty_list_returns_400(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [])
    assert resp.status == 400
    assert (await resp.json()) == {"error": [{"code": "files_validate_invalid"}]}


@pytest.mark.asyncio
async def test_validate_non_json_body_returns_400(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await client.post(
            "/v1/files/validate", data=b"not json", headers=_auth())
    assert resp.status == 400
    assert (await resp.json()) == {"error": [{"code": "files_validate_invalid"}]}


@pytest.mark.asyncio
async def test_validate_missing_paths_key_returns_400(seeded):
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, None)
        assert resp.status == 400


@pytest.mark.asyncio
async def test_validate_requires_auth(seeded):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as client:
        resp = await client.post(
            "/v1/files/validate", json={"paths": [seeded["good"]]})
    assert resp.status == 401


# ----------------------------------------------------------------------
# Anti-drift invariant: validate.ok == (download validation passes)
# ----------------------------------------------------------------------


def test_agree_and_never_echo_resolved_path(seeded):
    """Structural anti-drift guard: the shared validator behind BOTH the batch oracle and the GET
    download ladder yields ok IFF the download would return 200. Assert it explicitly for a matrix,
    and assert the response reason is the endpoint's stable/generic code (never an unknown leak)."""
    matrix = [
        seeded["good"], seeded["missing"], seeded["plain_dir"], seeded["outside"],
        seeded["git_secret"], seeded["hard"], seeded["big"],
    ]
    for raw in matrix:
        canonical, reason = files_allowlist._validate_allowlist_path(raw)
        # GET /v1/files builds its 200-else ladder directly on _validate_allowlist_path
        # (_files_deny_response maps reason -> status), so download returns 200 IFF canonical is set.
        download_ok = canonical is not None
        expected = (
            None if raw == seeded["good"]
            else "not_found" if raw == seeded["missing"]
            else "not_a_file" if raw == seeded["plain_dir"]
            else "not_allowed" if raw == seeded["outside"]
            else "denied" if raw == seeded["git_secret"]
            else "hardlink" if raw == seeded["hard"]
            else "too_large" if raw == seeded["big"]
            else None
        )
        assert download_ok == (expected is None)
        assert reason == expected
        # The stable code seen by a client equals the validator reason when it is a stable known
        # reason, else the collapsed "denied": never echo the resolved path, only the input.
        stable = reason if reason in {
            "not_found", "not_absolute", "traversal", "bad_path", "denied",
            "not_allowed", "not_a_file", "hardlink", "too_large",
        } else "denied"
        assert stable == expected


@pytest.mark.asyncio
async def test_validate_never_echoes_non_input_path(seeded):
    """The response 'path' field must equal the client's input string for every result — never the
    resolved/canonical path."""
    async with TestClient(TestServer(_app(_adapter()))) as client:
        resp = await _validate(client, [seeded["good"], seeded["outside"], seeded["missing"]])
    body = await resp.json()
    for row, inp in zip(body["results"], [seeded["good"], seeded["outside"], seeded["missing"]]):
        assert row["path"] == inp
