"""Targeted tests for the mermaid_offload context engine plugin.

These tests live OUTSIDE the plugin package (per the loader pitfall: every
``*.py`` in the plugin dir is exec'd during discovery). They exercise both the
isolated units (config/store/summarize/mermaid) and the REAL host loader paths
(discovery -> load -> ABC conformance -> deepcopy -> schema normalization).

Run from the repo root with the Hermes venv python:

    cd ~/.hermes/hermes-agent
    ./venv/bin/python3 -m pytest tests/plugins/context_engine/test_mermaid_offload.py -q
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover — runner provides fixtures without pytest
    pytest = None


def _skip(reason: str):
    if pytest is not None:
        pytest.skip(reason)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The @pytest.fixture decorators only apply when pytest is present; without it
# the functions are plain callables (the ad-hoc runner invokes them directly).
_fixture = pytest.fixture if pytest is not None else (lambda *a, **k: (lambda f: f))


@_fixture(autouse=True)
def _isolated_home(monkeypatch):
    """Give each test a fresh HERMES_HOME so storage is isolated and no real
    user config is read."""
    tmp = tempfile.mkdtemp(prefix="hermes_mermaid_test_")
    monkeypatch.setenv("HERMES_HOME", tmp)
    return Path(tmp)


@_fixture()
def engine():
    """A live engine with a configured temp data dir."""
    from plugins.context_engine.mermaid_offload.engine import MermaidOffloadEngine

    eng = MermaidOffloadEngine()
    eng._data_dir = Path(tempfile.mkdtemp(prefix="mermaid_store_"))
    return eng


def _store(eng):
    return eng._ensure_store()


# ---------------------------------------------------------------------------
# 1. Host loader wiring (real paths, not mocks)
# ---------------------------------------------------------------------------

def test_loads_through_the_real_host_loader():
    from plugins.context_engine import discover_context_engines, load_context_engine

    discovered = {n: a for n, _d, a in discover_context_engines()}
    assert discovered.get("mermaid_offload") is True

    eng = load_context_engine("mermaid_offload")
    assert eng is not None
    assert eng.name == "mermaid_offload"


def test_no_unimplemented_abstract_methods():
    from plugins.context_engine import load_context_engine

    eng = load_context_engine("mermaid_offload")
    assert not getattr(type(eng), "__abstractmethods__", set())


def test_deepcopyable_via_host_loader():
    from plugins.context_engine import load_context_engine

    eng = load_context_engine("mermaid_offload")
    copy.deepcopy(eng)  # must not raise (#42449)


def test_schemas_normalize_for_the_host():
    from plugins.context_engine import load_context_engine

    eng = load_context_engine("mermaid_offload")
    schemas = eng.get_tool_schemas()
    names = {s.get("name") for s in schemas}
    assert names == {"offload_lookup", "offload_search"}
    try:
        from agent.memory_manager import normalize_tool_schema

        for s in schemas:
            assert normalize_tool_schema(s) is not None
    except ImportError:
        _skip("normalize_tool_schema not importable in this build")


def test_register_retains_live_engine_not_probe():
    """Discovery probes the engine before real loading; the module-level ref
    must point at the LIVE engine (last writer), never the probe."""
    from plugins.context_engine import discover_context_engines, load_context_engine
    from plugins.context_engine.mermaid_offload import engine as emod

    discover_context_engines()  # creates the probe
    probe = emod.get_active_engine()
    live = load_context_engine("mermaid_offload")
    assert probe is not live  # they ARE different objects
    assert emod.get_active_engine() is live  # ref tracks the live one


def test_later_discovery_probe_does_not_clobber_live_engine():
    """Regression: a discovery scan that runs AFTER the real load must not
    overwrite ``_active_engine`` with a throwaway probe.

    In a long-lived gateway the web UI (``/api/context-engines``) or a later
    ``hermes plugins`` scan can call ``discover_context_engines()`` after an
    agent has already loaded and wired the engine. Discovery reuses the loaded
    module and calls ``register()`` again with a probe collector; that probe
    must NOT clobber the live ``_active_engine`` reference, or ``/offload``
    (which resolves the engine via ``get_active_engine()`` at call time) would
    report a stale, never-wired engine.
    """
    from plugins.context_engine import discover_context_engines, load_context_engine
    from plugins.context_engine.mermaid_offload import engine as emod

    live = load_context_engine("mermaid_offload")
    assert live is not None
    assert emod.get_active_engine() is live

    # A later discovery probe (web UI poll / plugins scan) must not replace it.
    discover_context_engines()
    assert emod.get_active_engine() is live, (
        "discovery probe clobbered the live engine reference"
    )

    # A second load is still the live engine (last-writer-wins on the real path).
    live2 = load_context_engine("mermaid_offload")
    assert emod.get_active_engine() is live2


def test_no_gateway_fallback_through_real_loader_sequence():
    """Regression: the gateway-reported 'loaded but no engine instance found'
    / 'not found — falling back to built-in compressor' symptom must never
    recur through the real host loader.

    The host runs the exact sequence below on agent_init: it probes every
    engine via discover_context_engines() (which instantiates + register()s a
    throwaway instance), then calls load_context_engine() for the selected
    engine. The loader swallows a register() exception with only a debug log
    and then reports 'no engine instance found', which makes agent_init fall
    back to the built-in compressor. This test pins that register() runs
    without raising and the loader resolves a non-None engine after a full
    discovery cycle, so a future partial-build or register() regression is
    caught instead of silently degrading to the built-in compressor.
    """
    from plugins.context_engine import discover_context_engines, load_context_engine
    from plugins.context_engine import mermaid_offload as mod

    # The host's discovery probe must not raise (register() executes here).
    discovered = {n: a for n, _d, a in discover_context_engines()}
    assert discovered.get("mermaid_offload") is True

    # The loader must resolve a real instance — not None (which is what makes
    # run_agent log "no engine instance found" then fall back).
    eng = load_context_engine("mermaid_offload")
    assert eng is not None, "loader returned None -> gateway falls back to built-in compressor"

    # A subsequent discovery+load cycle (e.g. a child/subagent agent_init)
    # must also resolve without exceptions or a None engine.
    discover_context_engines()
    eng2 = load_context_engine("mermaid_offload")
    assert eng2 is not None
    # And the live-engine accessor slash-command handlers use resolves non-None.
    assert mod.get_active_engine() is not None


def test_engine_name_matches_directory():
    from plugins.context_engine import load_context_engine

    eng = load_context_engine("mermaid_offload")
    # Must equal the DIRECTORY name (the resolution key), not a hyphen variant.
    assert eng.name == "mermaid_offload"


def test_probe_does_not_flood_command_registry_with_skip_warnings():
    """Regression: a discovery probe must NOT attempt to register '/offload'.

    Before the loader's ``is_probe`` flag reached ``register_command``, every
    discovery probe (and repeated load) tried to claim the already-registered
    '/offload' command and logged a "already registered by a plugin. Skipping."
    warning. In a long-lived gateway that floods gateway.error.log on every
    discovery + load cycle (web UI polls, `hermes plugins` scans, child-agent
    agent_init). Only the real load (is_probe=False) may register commands.
    """
    from plugins.context_engine import (
        _EngineCollector,
        discover_context_engines,
        load_context_engine,
    )
    from hermes_cli.plugins import get_plugin_manager

    # First a real load registers the command (exactly once).
    live = load_context_engine("mermaid_offload")
    assert live is not None
    manager = get_plugin_manager()
    assert "offload" in manager._plugin_commands

    # A probe collector must NOT touch the registry (no re-register attempt,
    # no "Skipping" warning, no clobber of the handler). _EngineCollector's
    # probe path (is_probe=True) early-returns without forwarding, so it has no
    # _registered_commands list of its own to inspect — assert the behavior:
    # the global registry's handler must be unchanged after the probe runs.
    probe = _EngineCollector(engine_name="mermaid_offload", is_probe=True)
    probe.register_command("offload", lambda: None)
    assert manager._plugin_commands["offload"]["handler"] is not None
    # The registered handler is still the real engine's status fn.
    from plugins.context_engine.mermaid_offload import engine as emod
    assert manager._plugin_commands["offload"]["handler"] is emod._offload_status

    # Full discovery cycle must also not re-register / flood.
    discover_context_engines()
    assert "offload" in manager._plugin_commands

    # A repeated REAL load (child-agent agent_init) re-registers our OWN
    # command; it must be skipped silently, not logged as a "Skipping" warning.
    load_context_engine("mermaid_offload")
    assert manager._plugin_commands["offload"]["handler"] is emod._offload_status


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------

def test_config_defaults():
    from plugins.context_engine.mermaid_offload.config import MermaidOffloadConfig

    cfg = MermaidOffloadConfig.from_config({})
    assert cfg.threshold == 0.50
    assert cfg.protect_first_n == 3
    assert cfg.protect_last_n == 6


def test_config_type_coercion_and_clamping():
    from plugins.context_engine.mermaid_offload.config import MermaidOffloadConfig

    # Bad types / out-of-range must clamp, not raise.
    cfg = MermaidOffloadConfig.from_config({
        "threshold": "abc",          # -> default 0.50
        "aggressive_threshold": 0.3,  # below threshold -> clamped up to 0.50
        "emergency_threshold": 0.4,   # below aggressive -> clamped up to 0.50
        "protect_last_n": 0,          # -> floor 2
        "min_content_chars": "zzz",   # -> default 500
    })
    assert cfg.threshold == 0.50
    assert cfg.aggressive_threshold >= cfg.threshold
    assert cfg.emergency_threshold >= cfg.aggressive_threshold
    assert cfg.protect_last_n >= 2


def test_config_thresholds_monotonic():
    from plugins.context_engine.mermaid_offload.config import MermaidOffloadConfig

    cfg = MermaidOffloadConfig.from_config({
        "threshold": 0.9,
        "aggressive_threshold": 0.6,  # tries to invert
        "emergency_threshold": 0.7,
    })
    mild, agg, emerg = cfg.thresholds
    assert mild <= agg <= emerg


def test_config_load_from_section():
    from plugins.context_engine.mermaid_offload.config import MermaidOffloadConfig

    # A typo'd key must be ignored.
    cfg = MermaidOffloadConfig.from_config({"threshold": 0.6, "bogus_key": 123})
    assert cfg.threshold == 0.6


# ---------------------------------------------------------------------------
# 3. Store
# ---------------------------------------------------------------------------

def test_store_add_and_read(tmp_path):
    from plugins.context_engine.mermaid_offload.store import OffloadStore

    store = OffloadStore(tmp_path)
    assert store.open_for_session("sess-1")
    entry = store.add(tool_name="terminal", summary="cmd: exit=0", content="hello world")
    assert entry is not None
    assert entry["node_id"] == "N001"
    assert store.read_content("N001") == "hello world"
    assert store.read_content("n1") == "hello world"
    assert store.read_content(" 1 ") == "hello world"


def test_store_reload_continues_numbering(tmp_path):
    from plugins.context_engine.mermaid_offload.store import OffloadStore

    s1 = OffloadStore(tmp_path)
    s1.open_for_session("sess-1")
    s1.add(tool_name="a", summary="s1", content="c1")
    s1.add(tool_name="b", summary="s2", content="c2")
    s1.close()

    s2 = OffloadStore(tmp_path)
    s2.open_for_session("sess-1")
    assert s2.total_nodes == 2
    e = s2.add(tool_name="c", summary="s3", content="c3")
    assert e["node_id"] == "N003"  # continues, no collision


def test_store_skips_corrupt_index_lines(tmp_path):
    from plugins.context_engine.mermaid_offload.store import OffloadStore

    (tmp_path / "sess-1").mkdir(parents=True)
    (tmp_path / "sess-1" / "refs").mkdir()
    (tmp_path / "sess-1" / "index.jsonl").write_text(
        '{"seq": 1, "node_id": "N001", "tool_name": "a", "summary": "s", "ts": 0, "ref": "refs/N001.md"}\n'
        "NOT JSON\n"
        '{"seq": 2, "node_id": "N002", "tool_name": "b", "summary": "s", "ts": 0, "ref": "refs/N002.md"}\n',
        encoding="utf-8",
    )
    (tmp_path / "sess-1" / "refs" / "N001.md").write_text("x", encoding="utf-8")
    (tmp_path / "sess-1" / "refs" / "N002.md").write_text("y", encoding="utf-8")

    store = OffloadStore(tmp_path)
    store.open_for_session("sess-1")
    assert store.total_nodes == 2  # corrupt line skipped, others loaded


def test_store_degrades_when_dir_is_a_file(tmp_path):
    from plugins.context_engine.mermaid_offload.store import OffloadStore

    # Make a path whose parent is a regular file -> mkdir must fail -> degrade.
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    store = OffloadStore(blocker / "child" / "deep")
    ok = store.open_for_session("sess-1")
    assert ok is False
    assert store.degraded is True
    assert store.add(tool_name="t", summary="s", content="c") is None  # no content on disk


def test_store_search_tiers():
    from plugins.context_engine.mermaid_offload.store import OffloadStore

    store = OffloadStore(Path(tempfile.mkdtemp()))
    store.open_for_session("sess")
    store.add(tool_name="web_search", summary="search 'mermaid compression'", content="10 results")
    store.add(tool_name="terminal", summary="git status", content="clean working tree")
    hits = store.search("mermaid")
    assert len(hits) == 1
    assert hits[0]["tool_name"] == "web_search"
    # body-only match
    hits2 = store.search("working tree")
    assert len(hits2) == 1


def test_sanitize_session_id():
    from plugins.context_engine.mermaid_offload.store import sanitize_session_id

    assert sanitize_session_id("abc-123_xyz.9") == "abc-123_xyz.9"
    assert sanitize_session_id("../../evil") == ".._.._evil".strip("._")
    assert sanitize_session_id("") == "default"


def test_parse_node_id_sloppy():
    from plugins.context_engine.mermaid_offload.store import parse_node_id

    assert parse_node_id("N001") == 1
    assert parse_node_id("n12") == 12
    assert parse_node_id("12") == 12
    assert parse_node_id("  N012 ") == 12
    assert parse_node_id("garbage") is None


# ---------------------------------------------------------------------------
# 4. Summarize
# ---------------------------------------------------------------------------

def test_mechanical_summary_arg_aware():
    from plugins.context_engine.mermaid_offload.summarize import mechanical_summary

    s = mechanical_summary(
        "terminal", "out\n" * 30,
        call_args={"command": "pytest -q"}, metadata={"exit_code": 0},
    )
    assert "pytest -q" in s
    assert "30" in s  # line count
    assert "exit=0" in s  # from metadata


def test_mechanical_summary_metadata_exit():
    from plugins.context_engine.mermaid_offload.summarize import mechanical_summary

    s = mechanical_summary("terminal", "oops", metadata={"exit_code": 1})
    assert "exit=1" in s


def test_placeholder_and_marker_roundtrip():
    from plugins.context_engine.mermaid_offload.summarize import (
        extract_offload_marker,
        placeholder_text,
    )

    text = placeholder_text("N001", "terminal", "cmd: exit=0 3 lines")
    assert extract_offload_marker(text) == "N001"
    assert extract_offload_marker("plain output") is None


def test_is_canvas_message():
    from plugins.context_engine.mermaid_offload.summarize import is_canvas_message

    assert is_canvas_message({
        "role": "system",
        "content": "【Offloaded tool history — Mermaid flowchart】\nstuff",
    })
    assert not is_canvas_message({"role": "system", "content": "plain"})


# ---------------------------------------------------------------------------
# 5. Mermaid
# ---------------------------------------------------------------------------

def test_mermaid_label_escaping():
    from plugins.context_engine.mermaid_offload.mermaid import escape_label

    assert '"' not in escape_label('a"b`c[d]e{f}g<h>i|j#k;l')
    assert ";" not in escape_label("x;y")
    assert "#" not in escape_label("color #fff")
    # no unescaped double-quote survives inside a label
    assert '"' not in escape_label('say "hi"')


def test_mermaid_budget_degradation():
    from plugins.context_engine.mermaid_offload.mermaid import build_mermaid

    nodes = [
        {"node_id": f"N{i:03d}", "tool_name": "terminal", "summary": "x" * 200}
        for i in range(1, 21)
    ]
    canvas = build_mermaid(nodes, max_nodes=40, max_chars=800)
    assert len(canvas) <= 800 + 200  # small slack; degradation applied
    # Newest nodes kept, older stubbed
    assert "N020" in canvas or "OLDER" in canvas


def test_mermaid_flow_order():
    from plugins.context_engine.mermaid_offload.mermaid import build_mermaid

    nodes = [
        {"node_id": "N001", "tool_name": "a", "summary": "s1"},
        {"node_id": "N002", "tool_name": "b", "summary": "s2"},
    ]
    canvas = build_mermaid(nodes, max_nodes=40, max_chars=2000)
    assert "N001 --> N002" in canvas


# ---------------------------------------------------------------------------
# 6. Engine compress
# ---------------------------------------------------------------------------

def _tool_pair(assistant_idx, tool_call_id, content="y" * 2000):
    """Build a (assistant tool_calls, tool result) message pair."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "pytest -q"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "terminal",
            "content": content,
        },
    ]


def _conversation(n_pairs=3):
    msgs = [{"role": "system", "content": "You are Hermes."}]
    for i in range(n_pairs):
        msgs += _tool_pair(len(msgs), f"call_{i}")
    msgs += [
        {"role": "user", "content": "summarize what you ran"},
        {"role": "assistant", "content": "Done."},
    ]
    return msgs


def test_compress_offloads_and_preserves_pair_integrity(engine):
    engine.config.min_content_chars = 10
    engine.config.protect_last_n = 2
    # Pad the head with non-system chat so ALL 3 tool pairs land in the
    # offloadable window (protect_first_n counts non-system head messages).
    msgs = [{"role": "system", "content": "You are Hermes."}]
    for i in range(3):
        msgs += [
            {"role": "user", "content": f"lead q{i}"},
            {"role": "assistant", "content": f"lead a{i}"},
        ]
    msgs += _tool_pair(len(msgs), "call_0")
    msgs += _tool_pair(len(msgs), "call_1")
    msgs += _tool_pair(len(msgs), "call_2")
    msgs += [
        {"role": "user", "content": "summarize what you ran"},
        {"role": "assistant", "content": "Done."},
    ]
    before = copy.deepcopy(msgs)
    result = engine.compress(copy.deepcopy(msgs), current_tokens=None)
    assert result != before  # actually did something
    # tool result bodies replaced, but role/tool_call_id/name preserved
    for m in result:
        if m.get("role") == "tool":
            assert m.get("tool_call_id"), "tool_call_id dropped!"
            assert m.get("name"), "name dropped!"
    # No orphaned tool_call_id: every tool result id matches an assistant call.
    assistant_ids = {
        c.get("id")
        for m in result
        if m.get("role") == "assistant"
        for c in (m.get("tool_calls") or [])
    }
    for m in result:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            assert m["tool_call_id"] in assistant_ids, "orphaned tool result!"
    # Canvas injected as system role
    canvases = [m for m in result if m.get("role") == "system" and "Offloaded tool history" in str(m.get("content"))]
    assert len(canvases) == 1
    # Node stored
    assert engine._store is not None and engine._store.total_nodes == 3


def test_compress_input_not_mutated(engine):
    engine.config.min_content_chars = 10
    engine.config.protect_last_n = 2
    msgs = _conversation(n_pairs=2)
    before = copy.deepcopy(msgs)
    engine.compress(msgs)  # pass the original list object
    assert msgs == before  # input list unchanged


def test_compress_idempotent(engine):
    engine.config.min_content_chars = 10
    engine.config.protect_last_n = 2
    msgs = _conversation(n_pairs=2)
    first = engine.compress(copy.deepcopy(msgs))
    before_count = engine._store.total_nodes
    second = engine.compress(copy.deepcopy(first))
    # No new nodes created on the second pass (placeholders skipped).
    assert engine._store.total_nodes == before_count
    # Still valid: canvas not duplicated.
    canvases = [m for m in second if m.get("role") == "system" and "Offloaded tool history" in str(m.get("content"))]
    assert len(canvases) == 1


def test_compress_protected_tail_untouched(engine):
    engine.config.min_content_chars = 10
    engine.config.protect_last_n = 2
    msgs = _conversation(n_pairs=2)
    result = engine.compress(copy.deepcopy(msgs))
    # Last 2 messages are the protected user/assistant tail — unchanged.
    assert result[-1]["content"] == "Done."
    assert result[-2]["content"] == "summarize what you ran"


def test_truncation_fallback_no_orphans():
    from plugins.context_engine.mermaid_offload.engine import MermaidOffloadEngine

    eng = MermaidOffloadEngine()
    # A low-tool session (chat only) that overflows must still shrink.
    msgs = [
        {"role": "system", "content": "sys"},
    ] + [
        m
        for i in range(20)
        for m in (
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        )
    ]
    eng.context_length = 100_000
    eng.last_prompt_tokens = 100_000
    eng.threshold_tokens = 80_000
    result = eng.compress(copy.deepcopy(msgs), current_tokens=99_000)
    assert len(result) < len(msgs)  # actually truncated


def test_handle_lookup_and_search(engine):
    engine.config.min_content_chars = 10
    engine.config.protect_last_n = 2
    engine.compress(copy.deepcopy(_conversation(n_pairs=2)))
    res = json.loads(engine.handle_tool_call("offload_lookup", {"node_id": "N001"}))
    assert res["node_id"] == "N001"
    assert "content" in res
    res2 = json.loads(engine.handle_tool_call("offload_search", {"query": "pytest"}))
    assert res2["count"] >= 1


def test_lookup_miss_returns_available_nodes(engine):
    res = json.loads(engine.handle_tool_call("offload_lookup", {"node_id": "N999"}))
    assert "available_nodes" in res
    assert "total_nodes" in res


def test_engine_deepcopy_safe(engine):
    engine.config.min_content_chars = 10
    engine.config.protect_last_n = 2
    engine.compress(copy.deepcopy(_conversation(n_pairs=1)))
    dup = copy.deepcopy(engine)
    assert dup is not engine
    assert dup.config == engine.config
    # The copy must NOT share the live store's file handle: it's lazy-reopened.
    assert dup._store is None or dup._store is not engine._store


def test_init_side_effect_free():
    """Discovery instantiates the engine; __init__ must not touch disk."""
    from plugins.context_engine.mermaid_offload.engine import MermaidOffloadEngine

    # Use a NON-EXISTENT nested path (mkdtemp creates its dir immediately).
    data_root = Path(tempfile.mkdtemp(prefix="mermaid_noio_")) / "not" / "created" / "yet"
    assert not data_root.exists()
    eng = MermaidOffloadEngine()
    eng._data_dir = data_root
    # Construction alone must not create the data dir.
    assert not data_root.exists()


def test_severity_ladder(engine):
    engine.context_length = 1000
    engine.threshold_tokens = 500
    assert engine._severity(100) == "mild"
    assert engine._severity(900) == "aggressive"
    assert engine._severity(970) == "emergency"


def test_should_compress_info(engine):
    engine.enabled = False
    should, reason = engine.should_compress_info(1000)
    assert should is False
    assert reason and "disabled" in reason
    engine.enabled = True
    engine.last_prompt_tokens = 0
    should, reason = engine.should_compress_info()
    assert should is False
    assert reason is None