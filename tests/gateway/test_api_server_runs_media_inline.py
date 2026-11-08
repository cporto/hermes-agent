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
