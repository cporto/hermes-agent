"""Mercury header context ring — the completion runtime must carry the model's
context window so the UI can render "used / total".

Mercury reads ``usage.runtime ?? event.runtime`` (App.tsx) and takes
``context_window`` from whichever wins, so BOTH must carry it:
  * default/global sessions get no usage.runtime -> the SSE event runtime must carry it
    (resolved from the model-metadata ladder; the run agent is already popped by the
    time _effective_turn_runtime runs).
  * model-locked/route ("gated") sessions attach usage.runtime, but its window came from
    the agent's compressor which is 0 when the agent ran without a resolved model, so
    _turn_runtime_metadata must fall back to the metadata ladder too.
The Desktop / TUI / CLI context meters read ``session.usage`` (next to ``session.stats``),
whose window comes from the agent's ``context_compressor.context_length`` -- same 0-when-
unresolved gap -- so gateway/session_usage._context falls back to metadata as well.
"""
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

import pytest


def _adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class _StubAgent:
    """Minimal agent: no resolved model, no compressor (the gated-session case)."""

    def __init__(self):
        self.context_compressor = None
        self.model = ""
        self.provider = ""
        self._hermes_api_runtime = {}


# --- event-runtime path (default/global sessions) -------------------------------------

@pytest.mark.asyncio
async def test_requested_model_surfaces_context_window():
    a = _adapter()
    rt = await a._effective_turn_runtime(
        {"requested": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
        {}, {})
    assert rt.get("context_window", 0) > 0


@pytest.mark.asyncio
async def test_default_session_falls_back_to_gateway_default_window(monkeypatch):
    from gateway import run
    monkeypatch.setattr(run, "_resolve_gateway_model",
                        lambda *a, **k: "deepseek-ai/DeepSeek-V4-Flash-0731")
    a = _adapter()
    rt = await a._effective_turn_runtime({"requested": {"provider": "", "model": ""}}, {}, {})
    assert rt.get("context_window", 0) > 0


@pytest.mark.asyncio
async def test_no_fabricated_compressor_threshold():
    a = _adapter()
    rt = await a._effective_turn_runtime(
        {"requested": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
        {}, {})
    assert "compressor_threshold_tokens" not in rt


@pytest.mark.asyncio
async def test_existing_window_is_not_overridden():
    a = _adapter()
    rt = await a._effective_turn_runtime(
        {"requested": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
        {"runtime": {"context_window": 4096}}, {})
    assert rt.get("context_window") == 4096


# --- usage.runtime path (gated/model-locked sessions, what Mercury prefers) ------------

def test_turn_runtime_metadata_surfaces_window_without_compressor():
    a = _adapter()
    rt = a._turn_runtime_metadata(
        _StubAgent(),
        route={"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        requested_runtime=None, route_source="session", confirmed_runtime_lock=False)
    assert rt.get("context_window", 0) > 0


# --- session.usage path (Desktop / TUI / CLI context meters) ---------------------------

class _StubCompressor:
    context_length = 0
    last_prompt_tokens = 1000
    _config_context_length = None
    compression_count = 0


def test_session_usage_context_falls_back_to_metadata(monkeypatch):
    from gateway import session_usage
    from gateway import run
    monkeypatch.setattr(run, "_resolve_gateway_model",
                        lambda *a, **k: "deepseek-ai/DeepSeek-V4-Flash-0731")

    class A:
        model = ""
        _session_messages = None
        _turn_base_usage_anchor = None

    r = session_usage._context(A(), _StubCompressor())
    assert (r.get("context_length") or 0) > 0


def test_session_usage_keeps_compressor_window_when_present():
    from gateway import session_usage

    class A:
        model = ""
        _session_messages = None
        _turn_base_usage_anchor = None

    class C:
        context_length = 4096
        last_prompt_tokens = 1000
        _config_context_length = None
        compression_count = 0

    r = session_usage._context(A(), C())
    assert r.get("context_length") == 4096
