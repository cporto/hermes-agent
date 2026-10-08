"""Mechanical (Sprint 1) summarization for offloaded tool output.

Sprint 1 is pure Python — no LLM, no network. Summaries are derived from the
tool name, the arguments the tool was called with, and cheap structural stats
of the output (line count, exit status). This is the lossy-but-usable layer
that makes the Mermaid map and retrieval index carry information.

Arg-aware summaries need the tool *arguments*, which live on the assistant
``tool_calls`` side of the pair — recovered in ``engine._infer_tool_call`` and
passed in here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Marker regex for our own placeholders so compress() is idempotent and never
# re-offloads content we already moved to disk.
OFFLOAD_MARKER_PREFIX = "📦 [offloaded:"
OFFLOAD_MARKER_RE = re.compile(r"^\s*📦\s*\[offloaded:\s*(N\d+)\]", re.MULTILINE)

# Subject keys, in priority order, used to build a short human label for a
# tool call (e.g. the command for terminal, the query for web_search).
_SUBJECT_KEYS = (
    "command",
    "query",
    "path",
    "file_path",
    "url",
    "pattern",
    "repo",
    "tool",
    "title",
    "name",
    "question",
    "symbol",
)


def extract_offload_marker(content: Any) -> Optional[str]:
    """Return the node id if ``content`` is one of our placeholders, else None.

    Accepts either a plain string or an OpenAI-format content list
    (``[{"type": "text", "text": ...}, ...]``). Returns ``None`` when content
    is not one of our offload placeholders.
    """
    text = _content_to_text(content)
    if not text:
        return None
    m = OFFLOAD_MARKER_RE.search(text)
    return m.group(1) if m else None


def is_canvas_message(msg: Dict[str, Any]) -> bool:
    """True if ``msg`` is our injected Mermaid canvas (system role + marker)."""
    if not isinstance(msg, dict):
        return False
    if msg.get("role") != "system":
        return False
    text = _content_to_text(msg.get("content"))
    return bool(text and "【Offloaded tool history — Mermaid flowchart】" in text)


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
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "input_text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block.get("text", "") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _extract_subject(tool_name: str, call_args: Dict[str, Any]) -> str:
    """Return a short subject label from the tool call arguments.

    Picks the first present key from a priority list; paths are tail-only so
    we don't blow the Mermaid label budget on long absolute paths.
    """
    if not call_args:
        return tool_name
    for key in _SUBJECT_KEYS:
        if key in call_args:
            val = call_args[key]
            if val is None:
                continue
            sval = str(val).strip()
            if not sval:
                continue
            # Tail-only for path-like values.
            if key in ("path", "file_path", "repo", "url"):
                parts = sval.replace("\\", "/").split("/")
                if len(parts) > 2:
                    sval = "/".join(parts[-2:])
            return sval[:80]
    return tool_name


def _text_stats(text: str) -> Dict[str, Any]:
    """Cheap structural stats: line count, byte-ish length, sample."""
    lines = text.splitlines()
    sample = next((ln.strip()[:120] for ln in lines if ln.strip()), "")
    return {
        "lines": len(lines),
        "chars": len(text),
        "sample": sample,
    }


def mechanical_summary(
    tool_name: str,
    text: str,
    call_args: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a mechanical one-line summary for a tool result.

    Examples:
        terminal(pytest -q): exit=0 30 lines
        web_search(python mermaid): 8 results, 1420 chars
        offload_lookup(N012): 512 chars, node ref
    """
    call_args = call_args or {}
    metadata = metadata or {}
    stats = _text_stats(text)
    lines = stats["lines"]
    chars = stats["chars"]

    subject = _extract_subject(tool_name, call_args)
    if subject != tool_name:
        base = f"{tool_name}({subject})"
    else:
        base = tool_name

    # Tool-specific hints from metadata where available.
    exit_code = metadata.get("exit_code")
    if exit_code is not None:
        return f"{base}: exit={exit_code} {lines} lines"

    # Generic fallback.
    return f"{base}: {lines} lines, {chars} chars"


def placeholder_text(node_id: str, tool_name: str, summary: str) -> str:
    """The compact replacement message body for an offloaded tool result."""
    return (
        f"📦 [offloaded:{node_id}] {tool_name}: {summary} "
        f"(full output in offload_lookup(node_id=\"{node_id}\"))"
    )


def canvas_preamble() -> str:
    """The teaching preamble placed above the Mermaid flowchart."""
    return (
        "【Offloaded tool history — Mermaid flowchart】\n"
        "Large tool outputs from earlier in this conversation were moved to "
        "disk to save context. Each node below is one offloaded tool call, in "
        "call order. To read the full raw output of a node, call "
        'offload_lookup(node_id="N001"). To find a node by keyword, call '
        'offload_search(query="..."). Do NOT re-run a tool to recover output '
        "that is already listed here."
    )


def condense_long_text(text: str, max_chars: int = 400) -> str:
    """Condense an oversized non-tool message body (emergency severity)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.6)].rstrip()
    tail = text[-int(max_chars * 0.3):].lstrip()
    return f"{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}"