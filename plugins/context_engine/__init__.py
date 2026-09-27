"""Context engine plugin discovery: ``plugins/context_engine/<name>/`` → ``ContextEngine``.
Engines ship in the repo, separate from the general plugin system; only one is active
(``context.engine`` in config.yaml; default ``"compressor"``, the built-in ContextCompressor)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from plugins import plugin_loader as _loader

logger = logging.getLogger(__name__)

_CONTEXT_ENGINE_PLUGINS_DIR = Path(__file__).parent


def discover_context_engines() -> List[Tuple[str, str, bool]]:
    """Return ``[(name, description, is_available), ...]`` for every bundled engine."""
    return [(child.name, _loader.read_plugin_description(child),
             _loader.probe_availability(lambda c=child: _load_engine_from_dir(c, is_probe=True)))
            for child in _loader.iter_plugin_dirs(_CONTEXT_ENGINE_PLUGINS_DIR)]


def load_context_engine(name: str) -> Optional["ContextEngine"]:  # noqa: F821
    """Load a ContextEngine instance by name; None if not found or it fails to load."""
    engine_dir = _CONTEXT_ENGINE_PLUGINS_DIR / name
    if not engine_dir.is_dir():
        logger.debug("Context engine '%s' not found in %s", name, _CONTEXT_ENGINE_PLUGINS_DIR)
        return None
    return _loader.load_named(
        name, engine_dir, _load_engine_from_dir, kind="Context engine", noun="engine", logger=logger
    )


def _load_engine_from_dir(engine_dir: Path, *, is_probe: bool = False) -> Optional["ContextEngine"]:  # noqa: F821
    """Import an engine module and extract its ContextEngine (register(ctx) or subclass).

    ``is_probe=True`` marks the call as coming from discovery (an availability probe),
    which may create a throwaway engine instance purely to check availability. Plugin
    ``register()`` entry points must NOT treat a probe as the live engine selection —
    see the mermaid_offload ``_active_engine`` retention contract (a probe must not
    clobber the live engine reference, and must not register slash commands).
    """
    from agent.context_engine import ContextEngine
    name = engine_dir.name
    mod = _loader.load_plugin_module(
        f"plugins.context_engine.{name}", engine_dir,
        parents=("plugins", "plugins.context_engine"), logger=logger)
    return mod and _loader.instance_from_module(
        mod, collector=_EngineCollector(engine_name=name, is_probe=is_probe), collected_attr="engine",
        base_cls=ContextEngine, name=name, logger=logger)


class _EngineCollector(_loader.NoopPluginContext):
    """Captures register_context_engine; forwards register_command to the global plugin command
    registry so engine slash commands behave like plugin ones."""

    def __init__(self, engine_name: str = "", is_probe: bool = False):
        self.engine = None
        self._engine_name = engine_name or "context_engine"
        # True when this collector is an availability probe from discovery (see
        # _load_engine_from_dir). Plugin register() entry points should not treat a
        # probe as the live engine selection.
        self.is_probe = is_probe

    def register_context_engine(self, engine):
        self.engine = engine

    def register_command(self, name: str, handler, description: str = "", args_hint: str = "") -> None:
        # Discovery probes are throwaway availability checks — a probe must NOT register
        # commands. Every probe would otherwise try (and fail, with an "already registered
        # by a plugin. Skipping." warning) to claim the same command, flooding the error
        # log on every discovery + load cycle in a long-lived gateway. The real load
        # (is_probe=False) registers the command exactly once.
        if self.is_probe:
            return
        clean = (name or "").lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning("Context engine '%s' tried to register a command with an empty name.",
                           self._engine_name)
            return
        conflict = "Context engine '%s' tried to register command '/%s' which %s Skipping."
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(conflict, self._engine_name, clean, "conflicts with a built-in command.")
                return
        except Exception:
            pass
        try:
            from hermes_cli.plugins import get_plugin_manager
            manager = get_plugin_manager()
            if clean in manager._plugin_commands:
                if manager._plugin_commands[clean].get("plugin") == f"context-engine:{self._engine_name}":
                    # Re-registration of OUR OWN command (repeated real load for a child
                    # agent, or a later load_context_engine call). Benign — first-writer-wins,
                    # handler resolves the live engine at call time — so skip silently rather
                    # than flooding the error log with "already registered" warnings.
                    return
                logger.warning(conflict, self._engine_name, clean, "is already registered by a plugin.")
                return
            manager._plugin_commands[clean] = {
                "handler": handler, "description": description or "Context engine command",
                "plugin": f"context-engine:{self._engine_name}", "args_hint": (args_hint or "").strip()}
            logger.debug("Context engine '%s' registered command: /%s", self._engine_name, clean)
        except Exception as exc:
            logger.debug("Context engine '%s' could not register /%s: %s", self._engine_name, clean, exc)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import importlib.util  # noqa: F401,E402
import sys  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
