"""Regression for #143666: /v1/runs finished output inlines MEDIA: image tags as
base64 data URLs, mirroring the other api_server output surfaces."""
import asyncio
import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter

# 1x1 transparent PNG (same bytes used by test_api_server_media_data_urls).
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.asyncio
async def test_runs_finished_output_inlines_media(adapter):
    d = Path(tempfile.mkdtemp(prefix="hermes_run_media"))
    p = d / "shot.png"
    p.write_bytes(_PNG_BYTES)

    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {
                "final_response": f"Here: MEDIA:{p}"
            }
            mock_agent.session_prompt_tokens = 10
            mock_agent.session_completion_tokens = 5
            mock_agent.session_total_tokens = 15
            mock_create.return_value = mock_agent

            resp = await cli.post("/v1/runs", json={"input": "hello"})
            assert resp.status == 202, await resp.text()
            run_id = (await resp.json())["run_id"]

            status = {}
            for _ in range(100):
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                if status.get("status") == "completed":
                    break
                await asyncio.sleep(0.05)
            assert status.get("status") == "completed", status
            out = status.get("output", "")
            assert "data:image/png;base64," in out, out
            assert "MEDIA:/" not in out, out

def test_run_projection_inlines_media(monkeypatch, tmp_path):
    """#143666 regression: the production GET /v1/runs/{id} output comes from the
    authority run_projection (which overrides the adapter-local _finish output), so
    it must inline MEDIA the same way. The first fix only touched _finish and left
    the raw path alive in production (unit test passed only because run_projection
    returns None without an authority admission)."""
    import gateway.platforms.api_server_authority_runs as AR

    p = tmp_path / "shot.png"
    p.write_bytes(_PNG_BYTES)

    class _DB:
        pass

    class _Auth:
        db = _DB()
        sessions = {}

    row = {"status": "terminal", "outcome": "completed", "admission_id": "a1",
           "target_session_id": "t1", "generation": 1}
    saved = {"result": {"final_response": f"Med: MEDIA:{p}", "status": "completed"},
             "usage": {}}

    monkeypatch.setattr(AR, "run_admission", lambda a, r: (_Auth(), row))
    monkeypatch.setattr(AR, "admission_result", lambda db, aid: saved)

    proj = AR.run_projection(None, "run_v1")
    assert proj is not None
    out = proj["output"]
    assert "data:image/png;base64," in out, out
    assert "MEDIA:/" not in out, out
