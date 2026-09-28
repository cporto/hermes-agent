"""Config resolution and validation for the mermaid_offload context engine.

The engine is configured from a TOP-LEVEL ``mermaid_offload:`` section in
config.yaml (mirroring how the built-in compressor reads ``context:``). Read
via ``cfg_get(load_config_readonly(), "mermaid_offload", default={})``.

Every value is coerced to the type of its default and clamped so a typo'd
config can never raise during ``__init__`` or invert the severity ladder.
Unknown keys are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from hermes_cli.config import cfg_get, load_config_readonly


# Defaults mirrored in plugin.yaml. Keep both in sync.
DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "threshold": 0.50,
    "aggressive_threshold": 0.85,
    "emergency_threshold": 0.95,
    "min_content_chars": 500,
    "protect_first_n": 3,
    "protect_last_n": 6,
    "canvas_max_nodes": 40,
    "canvas_max_chars": 2000,
    "lookup_max_chars": 10000,
    "retention_days": 14,
}


@dataclass
class MermaidOffloadConfig:
    """Validated engine configuration.

    Attributes mirror the ``mermaid_offload:`` config section after type
    coercion and clamping. The three thresholds are forced monotonic so a bad
    config cannot invert the severity ladder
    (mild <= aggressive <= emergency).
    """

    enabled: bool = True
    threshold: float = 0.50
    aggressive_threshold: float = 0.85
    emergency_threshold: float = 0.95
    min_content_chars: int = 500
    protect_first_n: int = 3
    protect_last_n: int = 6
    canvas_max_nodes: int = 40
    canvas_max_chars: int = 2000
    lookup_max_chars: int = 10000
    retention_days: int = 14

    # Derived / runtime fields, not read from config.
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_config(
        cls, section: Dict[str, Any] | None = None
    ) -> "MermaidOffloadConfig":
        """Build a validated config from a (possibly partial) dict.

        ``section`` may be ``None`` (no config section present) — defaults are
        used. This is called from ``__init__`` so it MUST be cheap and pure;
        discovery instantiates the engine even when it is not the selected
        engine.
        """
        section = section or {}

        def _f(key: str, default: float) -> float:
            val = section.get(key, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        def _i(key: str, default: int) -> int:
            val = section.get(key, default)
            try:
                return int(val)
            except (TypeError, ValueError):
                return int(default)

        def _b(key: str, default: bool) -> bool:
            val = section.get(key, default)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in ("1", "true", "yes", "on")
            return bool(val)

        threshold = min(max(_f("threshold", DEFAULT_CONFIG["threshold"]), 0.05), 0.99)
        # Enforce monotonic severity ladder; clamp within (0, 1).
        aggressive = min(
            max(_f("aggressive_threshold", DEFAULT_CONFIG["aggressive_threshold"]), threshold),
            0.995,
        )
        emergency = min(
            max(
                _f("emergency_threshold", DEFAULT_CONFIG["emergency_threshold"]),
                aggressive,
            ),
            0.999,
        )

        # Floor protect_last_n at 2 so we never offload the tool result the
        # agent is actively reading this turn.
        protect_last_n = max(_i("protect_last_n", DEFAULT_CONFIG["protect_last_n"]), 2)

        return cls(
            enabled=_b("enabled", DEFAULT_CONFIG["enabled"]),
            threshold=threshold,
            aggressive_threshold=aggressive,
            emergency_threshold=emergency,
            min_content_chars=max(_i("min_content_chars", DEFAULT_CONFIG["min_content_chars"]), 0),
            protect_first_n=max(_i("protect_first_n", DEFAULT_CONFIG["protect_first_n"]), 0),
            protect_last_n=protect_last_n,
            canvas_max_nodes=max(_i("canvas_max_nodes", DEFAULT_CONFIG["canvas_max_nodes"]), 1),
            canvas_max_chars=max(_i("canvas_max_chars", DEFAULT_CONFIG["canvas_max_chars"]), 200),
            lookup_max_chars=max(_i("lookup_max_chars", DEFAULT_CONFIG["lookup_max_chars"]), 500),
            retention_days=max(_i("retention_days", DEFAULT_CONFIG["retention_days"]), 0),
            _raw=dict(section),
        )

    @classmethod
    def load(cls) -> "MermaidOffloadConfig":
        """Load config from the active config.yaml ``mermaid_offload`` section."""
        try:
            section = cfg_get(load_config_readonly(), "mermaid_offload", default={}) or {}
        except Exception:
            section = {}
        return cls.from_config(section)

    @property
    def thresholds(self) -> "tuple[float, float, float]":
        """(mild, aggressive, emergency) severity thresholds, monotonic."""
        return (self.threshold, self.aggressive_threshold, self.emergency_threshold)