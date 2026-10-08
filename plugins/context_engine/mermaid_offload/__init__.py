"""mermaid_offload context engine plugin entry point.

Registration uses the preferred ``register(ctx)`` pattern. Discovery
instantiates a throwaway PROBE engine before real loading (pitfall 0c), and
the host's command registry is first-writer-wins — so slash-command handlers
MUST resolve the live engine at CALL time through ``get_active_engine()`` in
``engine.py``, never close over an instance captured here. ``load_context_engine``
runs after discovery, so last-writer-wins lands on the real engine.
"""

from __future__ import annotations

from typing import Any

from . import engine as _engine_mod
from .engine import MermaidOffloadEngine, get_active_engine

__all__ = [
    "MermaidOffloadEngine",
    "register",
    "get_active_engine",
]


def register(ctx: Any) -> None:
    """Entry point called by both the discovery collector and the real loader.

    ``ctx`` exposes ``register_context_engine`` and (optionally)
    ``register_command``. One implementation serves both paths. The engine is
    constructed with ZERO args (the fallback loader path uses ``attr()`` and
    would reject a required-arg constructor).

    Discovery probes availability by calling ``register()`` with a throwaway
    collector (``ctx.is_probe`` is True). A probe MUST NOT overwrite the live
    ``_active_engine`` reference: discovery can run after a real load (e.g. a
    later ``hermes plugins`` scan or a web-UI context-engine poll), and if a
    probe clobbered the ref, ``/offload`` would resolve a stale, never-wired
    engine. Only the real ``load_context_engine`` path (``is_probe`` False)
    updates ``_active_engine``.
    """
    engine = MermaidOffloadEngine()
    ctx.register_context_engine(engine)
    # Retain the LIVE engine so slash-command handlers resolve it at call time.
    # Assign through the engine MODULE namespace (not a rebinding local
    # import), so get_active_engine() in engine.py observes the update. Only
    # the real load (not a discovery probe) may set this.
    if not getattr(ctx, "is_probe", False):
        _engine_mod._active_engine = engine

    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):
        register_command(
            "offload",
            _engine_mod._offload_status,
            description="Show mermaid_offload engine / offload store status",
            args_hint="",
        )