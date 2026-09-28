"""MermaidOffloadEngine — a mechanical offload/replace context engine.

Sprint 1 is pure Python (no LLM, no network). Large tool outputs are moved to
disk (an append-only store per session) and replaced in the transcript with a
compact placeholder plus a Mermaid flowchart canvas that maps what was
offloaded. Retrieval tools ``offload_lookup`` / ``offload_search`` let the
agent recover full output on demand.

Key correctness properties (see the reference implementation):
  * ``register()`` retains the LIVE engine via a module-level ref because
    discovery instantiates the engine before real loading (pitfall 0c).
  * Pair integrity: tool result placeholders are rewritten IN PLACE preserving
    ``tool_call_id`` / ``name`` so the host sanitizer never orphans a pair.
  * The injected canvas uses ``role: "system"`` (alternation-neutral), is
    inserted at a stable offset right after the protected head, and replaces
    the prior canvas rather than stacking.
  * Idempotent: never re-offloads our own placeholders.
  * Storage degrades, never raises; a placeholder is only written once the ref
    file is confirmed on disk.
  * ``__init__`` is side-effect free (discovery instantiates it).
  * Deepcopyable: the store copies data only, never the file handle / lock.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from agent.context_engine import ContextEngine

from .config import MermaidOffloadConfig
from .mermaid import build_mermaid
from .store import OffloadStore, format_node_id, parse_node_id
from .summarize import (
    canvas_preamble,
    condense_long_text,
    extract_offload_marker,
    is_canvas_message,
    mechanical_summary,
    placeholder_text,
)

logger = logging.getLogger(__name__)

ENGINE_NAME = "mermaid_offload"

# Tool schemas (OpenAI format) the engine exposes to the agent.
_TOOL_LOOKUP_SCHEMA = {
    "type": "function",
    "name": "offload_lookup",
    "description": (
        "Retrieve the full raw output of an offloaded tool call by node id "
        '(e.g. "N001"). The output was moved to disk to save context; this '
        "restores it. Do NOT re-run the tool to recover output that is "
        "already captured here."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": 'Offloaded node id, e.g. "N001".',
            },
            "max_chars": {
                "type": "integer",
                "description": "Cap returned content length (default 10000).",
            },
        },
        "required": ["node_id"],
    },
}

_TOOL_SEARCH_SCHEMA = {
    "type": "function",
    "name": "offload_search",
    "description": (
        "Search offloaded tool outputs by keyword. Returns matching nodes "
        "with their summaries and truncated content. Use this to find an "
        "offloaded result without re-running the tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword to search summaries and output bodies.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 8).",
            },
        },
        "required": ["query"],
    },
}


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI content (str or list of content blocks) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


class MermaidOffloadEngine(ContextEngine):
    """Offload/replace context engine backed by per-session disk storage."""

    # -- Identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return ENGINE_NAME

    # -- Construction -----------------------------------------------------

    def __init__(self, config: Optional[MermaidOffloadConfig] = None):
        # Cheap, pure — NEVER do I/O here (discovery instantiates us).
        self.config = config or MermaidOffloadConfig.load()
        self.enabled = self.config.enabled

        # Token state (read by run_agent.py).
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.threshold_tokens: int = 0
        self.context_length: int = 0
        self.compression_count: int = 0
        self.threshold_percent: float = self.config.threshold

        # Head/tail protection.
        self.protect_first_n: int = self.config.protect_first_n
        self.protect_last_n: int = self.config.protect_last_n

        # Silence routine automatic compaction — ours is cheap & lossless-ish.
        self.emit_automatic_compaction_status: bool = False

        # Lazy per-session store.
        self._store: Optional[OffloadStore] = None
        self._data_dir = _resolve_data_dir()
        self._session_id: Optional[str] = None
        self._session_dir_created = False
        self._store_degraded_reason: Optional[str] = None

        # A plain RLock survives deepcopy poorly across threads; keep the lock
        # OUT of copied state by creating it lazily (see __deepcopy__).
        self._lock = threading.RLock()
        self._deepcopy_friendly = True

    # -- Config helper ----------------------------------------------------

    def _severity(self, current_tokens: Optional[int]) -> str:
        """Return 'mild' | 'aggressive' | 'emergency' for this turn."""
        if current_tokens is None or self.context_length <= 0:
            prompt = max(self.last_prompt_tokens, 0)
            if prompt <= 0:
                return "mild"
            ratio = prompt / self.context_length
        else:
            ratio = current_tokens / self.context_length
        if ratio >= self.config.emergency_threshold:
            return "emergency"
        if ratio >= self.config.aggressive_threshold:
            return "aggressive"
        if ratio >= self.config.threshold:
            return "mild"
        return "mild"

    # -- Store lifecycle --------------------------------------------------

    def _ensure_store(self) -> Optional[OffloadStore]:
        """Return the session store, opening it lazily on first use."""
        if self._store is not None:
            return self._store
        sid = self._session_id or "default"
        try:
            store = OffloadStore(self._data_dir)
            if store.open_for_session(sid):
                self._store = store
                self._session_dir_created = True
                return store
            self._store_degraded_reason = (
                f"storage unavailable ({sid}): {store._degraded_reason}"
            )
            return None
        except Exception as exc:  # noqa: BLE001 — degrade, never raise
            self._store_degraded_reason = f"storage error: {exc}"
            logger.warning("mermaid_offload store init failed: %s", exc)
            return None

    # -- ABC: token tracking ----------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        usage = usage or {}
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if prompt_tokens is not None and prompt_tokens >= 0:
            self.last_prompt_tokens = prompt_tokens
        if not self.enabled:
            return False
        if self.last_prompt_tokens <= 0:
            return False
        return self.last_prompt_tokens >= self.threshold_tokens

    def should_compress_info(self, prompt_tokens=None) -> Tuple[bool, Optional[str]]:
        if not self.enabled:
            return False, "mermaid_offload disabled in config"
        if self._store_degraded_reason:
            return False, f"offload storage degraded: {self._store_degraded_reason}"
        return self.should_compress(prompt_tokens), None

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        # Cheap chars/4 estimate so we can fire compression before the API call
        # when clearly over budget.
        total_chars = sum(len(_content_to_text(m.get("content"))) for m in messages)
        return total_chars / 4 >= self.threshold_tokens

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        head_end, tail_start = self._protected_bounds(messages)
        for i in range(head_end, tail_start):
            msg = messages[i]
            if self._is_offloadable_tool_message(msg, 1):
                return True
        return False

    def update_model(self, model="", context_length=0, base_url="", api_key="",
                     provider="", api_mode=""):
        try:
            super().update_model(model=model, context_length=context_length,
                                 base_url=base_url, api_key=api_key,
                                 provider=provider, api_mode=api_mode)
        except Exception:  # noqa: BLE001 — keep usable regardless
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        store = self._store
        status.update({
            "engine": self.name,
            "enabled": self.enabled,
            "nodes": len(store) if store else 0,
            "store_degraded": bool(self._store_degraded_reason),
            "config_threshold": self.config.threshold,
        })
        return status

    # -- ABC: lifecycle ---------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._store = None  # force reload from disk
        store = self._ensure_store()
        if store is not None:
            # Periodic sweep of orphaned session dirs (best-effort).
            try:
                store.sweep(self.config.retention_days)
            except Exception:  # noqa: BLE001
                pass

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        store = self._store
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        self._store = None
        self._session_dir_created = False

    def on_session_reset(self) -> None:
        super().on_session_reset()
        store = self._store
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        self._store = None
        self._session_dir_created = False

    def on_turn_complete(self, messages, usage=None, **kwargs) -> None:
        # Best-effort: persist the latest canvas after a finished turn so the
        # on-disk map tracks the in-memory one.
        store = self._ensure_store()
        if store is not None and len(store):
            try:
                nodes = store.available_nodes(limit=self.config.canvas_max_nodes)
                canvas = build_mermaid(
                    nodes,
                    max_nodes=self.config.canvas_max_nodes,
                    max_chars=self.config.canvas_max_chars,
                    preamble=canvas_preamble(),
                )
                store.write_canvas(canvas)
            except Exception:  # noqa: BLE001
                pass

    # -- ABC: protected bounds / helpers ----------------------------------

    def _protected_bounds(self, messages: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Return (head_end, tail_start); offloadable window is [head_end:tail_start).

        ``protect_first_n`` counts NON-SYSTEM head messages; all leading system
        messages (prompt + any prior canvas) are skipped first, so we never
        spend head budget on the system prompt and stay idempotent.
        """
        total = len(messages)
        i = 0
        while i < total and isinstance(messages[i], dict) and messages[i].get("role") == "system":
            i += 1
        head_end = min(i + self.config.protect_first_n, total)
        tail_start = max(total - self.config.protect_last_n, head_end)
        return head_end, tail_start

    def _is_offloadable_tool_message(self, msg: Dict[str, Any], min_chars: int) -> bool:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") != "tool":
            return False
        text = _content_to_text(msg.get("content"))
        if len(text) < min_chars:
            return False
        # Never re-offload our own placeholders (idempotency).
        if extract_offload_marker(msg.get("content")) is not None:
            return False
        return True

    @staticmethod
    def _is_tool_message(msg: Dict[str, Any]) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool"

    @staticmethod
    def _has_tool_calls(msg: Dict[str, Any]) -> bool:
        if not isinstance(msg, dict):
            return False
        tc = msg.get("tool_calls")
        return bool(tc)

    def _infer_tool_call(self, messages: List[Dict[str, Any]], index: int) -> Tuple[str, Dict[str, Any]]:
        """Walk BACK to the assistant tool_call matching this result's id.

        Returns ``(name, decoded_arguments)``. Matches ids via ``call_id or
        id`` (Codex sets id != call_id). Handles dict AND SimpleNamespace
        ``tool_calls``; ``arguments`` may be a dict or a JSON string.
        """
        result = messages[index]
        want = result.get("tool_call_id") or result.get("id")
        for j in range(index - 1, -1, -1):
            msg = messages[j]
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "tool":
                continue
            for call in (msg.get("tool_calls") or []):
                call_id = None
                args = None
                name = None
                if isinstance(call, dict):
                    call_id = call.get("id") or call.get("call_id")
                    # Standard OpenAI tool_calls nest arguments under
                    # function.arguments; fall back to a top-level
                    # "arguments" key for non-standard shapes.
                    args = call.get("arguments") or (
                        call.get("function", {}).get("arguments")
                    )
                    name = call.get("name") or call.get("function", {}).get("name")
                else:  # SimpleNamespace / object
                    call_id = getattr(call, "id", None) or getattr(call, "call_id", None)
                    name = getattr(call, "name", None)
                    fn = getattr(call, "function", None)
                    if fn is not None:
                        name = getattr(fn, "name", name)
                        args = getattr(fn, "arguments", None)
                if want and call_id and str(call_id) == str(want):
                    return _coerce_name(name), _decode_args(args)
        return "tool", {}

    # -- ABC: compress (the 3-pass) ---------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return list(messages)

        severity = self._severity(current_tokens)
        # Work on a fresh shallow copy so we never mutate the input list.
        working = [dict(m) if isinstance(m, dict) else m for m in messages]

        # Drop the PREVIOUS canvas first (prevents stacking; keeps stale
        # metadata from consuming head protection).
        working = [
            m for m in working
            if not (isinstance(m, dict) and is_canvas_message(m))
        ]

        head_end, tail_start = self._protected_bounds(working)
        min_chars = self.config.min_content_chars if severity == "mild" else 1

        store = self._ensure_store()
        offloaded = 0
        if store is not None:
            for i in range(head_end, tail_start):
                msg = working[i]
                if not self._is_offloadable_tool_message(msg, min_chars):
                    continue
                text = _content_to_text(msg.get("content"))
                tool_name, call_args = self._infer_tool_call(working, i)
                metadata = {
                    "tool_call_id": msg.get("tool_call_id") or msg.get("id"),
                }
                summary = mechanical_summary(tool_name, text, call_args=call_args,
                                             metadata=metadata)
                entry = store.add(
                    tool_name=tool_name,
                    summary=summary,
                    content=text,
                    placeholder_hint=json.dumps(call_args, ensure_ascii=False)[:200],
                    metadata=metadata,
                )
                if entry is None:
                    break  # write failed: NEVER point at content not on disk
                # Rewrite IN PLACE — preserve tool_call_id, name, role.
                new_msg = dict(msg)
                new_msg["content"] = placeholder_text(
                    entry["node_id"], tool_name, summary
                )
                working[i] = new_msg
                offloaded += 1

        # Emergency: condense oversized non-tool bodies too.
        if severity == "emergency":
            for i in range(head_end, tail_start):
                msg = working[i]
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") in ("user", "assistant"):
                    text = _content_to_text(msg.get("content"))
                    if len(text) > 2000 and extract_offload_marker(text) is None:
                        new_msg = dict(msg)
                        new_msg["content"] = condense_long_text(text)
                        working[i] = new_msg

        # Inject canvas right AFTER the protected head (stable offset).
        if store is not None and len(store):
            nodes = store.available_nodes(limit=self.config.canvas_max_nodes)
            canvas_text = build_mermaid(
                nodes,
                max_nodes=self.config.canvas_max_nodes,
                max_chars=self.config.canvas_max_chars,
                preamble=canvas_preamble(),
            )
            working.insert(head_end, {
                "role": "system",
                "content": canvas_text,
            })

        # If nothing was offloaded and we're past mild, truncate whole
        # assistant+tool groups so low-tool sessions don't overflow forever.
        if offloaded == 0 and severity != "mild":
            working = self._truncation_fallback(working, severity)

        self.compression_count += 1
        self.last_prompt_tokens = 0  # "awaiting real usage" — don't re-fire
        return working

    def _truncation_fallback(self, messages: List[Dict[str, Any]], severity: str) -> List[Dict[str, Any]]:
        """Drop whole assistant+tool groups so pairing survives (no orphans)."""
        head_end, tail_start = self._protected_bounds(messages)
        # Keep roughly 60% of the window on aggressive, 45% on emergency.
        keep_ratio = 0.6 if severity == "aggressive" else 0.45
        window = messages[head_end:tail_start]
        target_drop = max(0, len(window) - int(len(window) * keep_ratio))
        if target_drop <= 0:
            return messages

        # Build groups: [assistant_idx, tool_idx, tool_idx...].
        groups = []
        i = head_end
        while i < tail_start:
            group = [i]
            if self._has_tool_calls(messages[i]):
                j = i + 1
                while j < tail_start and self._is_tool_message(messages[j]):
                    group.append(j)
                    j += 1
                i = j
            else:
                i += 1
            groups.append(group)

        to_drop = set()
        # Drop from the OLDEST groups first (they're furthest from current work).
        for group in groups:
            if len(to_drop) >= target_drop:
                break
            first = messages[group[0]]
            if self._is_tool_message(first) and len(group) == 1:
                continue  # never drop a lone tool result — orphans it
            to_drop.update(group)

        result = [
            m for idx, m in enumerate(messages)
            if idx not in to_drop
        ]
        return result

    # -- ABC: tools -------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(_TOOL_LOOKUP_SCHEMA), copy.deepcopy(_TOOL_SEARCH_SCHEMA)]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        store = self._ensure_store()
        try:
            if name == "offload_lookup":
                node_id = str(args.get("node_id", ""))
                max_chars = int(args.get("max_chars") or self.config.lookup_max_chars)
                return self._handle_lookup(store, node_id, max_chars)
            if name == "offload_search":
                query = str(args.get("query", ""))
                limit = int(args.get("limit") or 8)
                return self._handle_search(store, query, limit)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        return json.dumps({"error": f"Unknown mermaid_offload tool: {name}"})

    def _handle_lookup(self, store: Optional[OffloadStore], node_id: str, max_chars: int) -> str:
        if store is None:
            return json.dumps({"error": "offload storage unavailable",
                               "hint": "retry later; storage degraded"})
        seq = parse_node_id(node_id)
        if seq is None:
            return json.dumps({"error": f"unrecognized node id {node_id!r}"})
        node = store.read_node(format_node_id(seq))
        if node is None:
            return json.dumps({
                "error": f"node {format_node_id(seq)} not found",
                "available_nodes": [
                    {"node_id": n["node_id"], "tool_name": n["tool_name"],
                     "summary": n["summary"]}
                    for n in store.available_nodes(10)
                ],
                "total_nodes": store.total_nodes,
            })
        content = node.get("content", "")
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True
        return json.dumps({
            "node_id": node["node_id"],
            "tool_name": node["tool_name"],
            "summary": node["summary"],
            "ref": node.get("ref"),
            "truncated": truncated,
            "content": content,
        })

    def _handle_search(self, store: Optional[OffloadStore], query: str, limit: int) -> str:
        if store is None:
            return json.dumps({"error": "offload storage unavailable"})
        limit = min(max(int(limit), 1), 25)
        results = store.search(query, limit=limit, max_chars=self.config.lookup_max_chars)
        return json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        })

    # -- Deepcopy support (#42449) ---------------------------------------

    def __deepcopy__(self, memo: Dict[int, Any]) -> "MermaidOffloadEngine":
        """Copy budget/state only — never the store's file handle or lock.

        agent_init deep-copies the engine per child agent. If deepcopy raised,
        Hermes silently falls back to the built-in compressor. We build a fresh
        engine with the same config and token state, and leave the store to be
        lazily reopened (the copy must not share the open index handle).
        """
        new = MermaidOffloadEngine(config=self.config)
        new.last_prompt_tokens = self.last_prompt_tokens
        new.last_completion_tokens = self.last_completion_tokens
        new.last_total_tokens = self.last_total_tokens
        new.threshold_tokens = self.threshold_tokens
        new.context_length = self.context_length
        new.compression_count = self.compression_count
        new.threshold_percent = self.threshold_percent
        new.protect_first_n = self.protect_first_n
        new.protect_last_n = self.protect_last_n
        new._session_id = self._session_id
        new._data_dir = self._data_dir
        new._session_dir_created = self._session_dir_created
        new._store_degraded_reason = self._store_degraded_reason
        # Do NOT copy self._store (holds an open file handle + RLock).
        new._store = None
        memo[id(self)] = new
        return new


# -- module-level helpers --------------------------------------------------

def _resolve_data_dir() -> Any:
    """Resolve the on-disk storage root without doing I/O.

    Stored under ``<config_dir>/context_engine/mermaid_offload``. Resolution
    must be cheap and pure (called from ``__init__`` during discovery).
    """
    from hermes_cli.config import get_config_path

    try:
        base = get_config_path()
    except Exception:  # noqa: BLE001 — fall back to a stable default
        from pathlib import Path

        base = Path.home() / ".hermes" / "config.yaml"
    return base.parent / "context_engine" / "mermaid_offload"


def _coerce_name(name: Any) -> str:
    if name is None:
        return "tool"
    return str(name)


def _decode_args(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            pass
        return {"raw": args[:500]}
    return {}


# -- live-engine retention for command handlers ---------------------------

_active_engine: Optional[MermaidOffloadEngine] = None


def get_active_engine() -> Optional[MermaidOffloadEngine]:
    """Return the LIVE engine (last registered), for slash-command handlers.

    Discovery instantiates a probe engine before real loading, and the host's
    command registry is first-writer-wins — so handlers must resolve the live
    instance at CALL time, never close over an instance at register time.
    ``load_context_engine()`` runs after discovery, so last-writer-wins lands
    on the real engine.
    """
    return _active_engine


def _offload_status(args: str = "") -> str:
    engine = get_active_engine()
    if engine is None:
        return "mermaid_offload: not active in this session."
    store = engine._store
    status = engine.get_status()
    lines = [
        f"mermaid_offload engine: {status.get('engine')}",
        f"  enabled: {status.get('enabled')}",
        f"  compression_count: {status.get('compression_count')}",
        f"  offloaded nodes: {len(store) if store else 0}",
        f"  threshold: {engine.config.threshold}",
        f"  store_degraded: {status.get('store_degraded')}",
    ]
    if store is not None and len(store):
        recent = store.available_nodes(5)
        if recent:
            lines.append("  recent nodes:")
            for n in recent:
                lines.append(f"    {n['node_id']} {n['tool_name']}: {n['summary']}")
    return "\n".join(lines)