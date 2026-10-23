"""Mercury header context ring — the SSE completion runtime must carry the model's
context window so the UI can render "used / total".

Regression guard for the fix: ``_effective_turn_runtime`` must surface
``context_window`` even for default/global sessions (where ``_finish_turn_result``
intentionally attaches no runtime dict, per the runtime-absence contract in
test_session_api.py). It resolves the window from the model-metadata ladder
because ``_run_agent``'s finally pops the run agent before this runs.
"""
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def test_requested_model_surfaces_context_window():
    a = _adapter()
    rt = a._effective_turn_runtime(
        {"requested": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
        {}, {})
    assert rt.get("context_window", 0) > 0


def test_default_session_falls_back_to_gateway_default_window(monkeypatch):
    # A client that sends no model relies on the gateway default model. Pin the
    # resolver so the test is deterministic (the isolated test env has no config).
    import gateway.run as run
    monkeypatch.setattr(run, "_resolve_gateway_model",
                        lambda *a, **k: "deepseek-ai/DeepSeek-V4-Flash-0731")
    a = _adapter()
    rt = a._effective_turn_runtime({"requested": {"provider": "", "model": ""}}, {}, {})
    assert rt.get("context_window", 0) > 0


def test_no_fabricated_compressor_threshold():
    # Only the window is known from metadata, not the real compression threshold,
    # so the runtime must not carry a made-up compressor_threshold_tokens.
    a = _adapter()
    rt = a._effective_turn_runtime(
        {"requested": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
        {}, {})
    assert "compressor_threshold_tokens" not in rt


def test_existing_window_is_not_overridden():
    a = _adapter()
    rt = a._effective_turn_runtime(
        {"requested": {"provider": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
        {"runtime": {"context_window": 4096}}, {})
    assert rt.get("context_window") == 4096
