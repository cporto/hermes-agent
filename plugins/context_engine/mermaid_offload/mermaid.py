"""Mermaid flowchart generation for the offload canvas.

The canvas is a ``flowchart LR`` graph where each node is one offloaded tool
call, in call order. Edges are call-order only — that is the sole relation
knowable mechanically; inferring real data flow is the LLM phase's job.

Label escaping is not optional: ``"``, backtick, ``[ ] { } < > |`` break
flowchart parsing; ``#`` starts an entity reference and ``;`` terminates one.
We strip/replace all of them, collapse whitespace, and cap the label length.

Budget degradation ladder over ``canvas_max_chars``: drop subgraphs first,
then halve the node count until it fits. Keep the NEWEST nodes and emit an
``OLDER[...]`` stub pointing at ``offload_search``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_MERMAUD_ESCAPE_RE = re.compile(r'[`"\[\]{}<>|#;]')


def escape_label(text: str, max_len: int = 90) -> str:
    """Escape a string for safe use inside a Mermaid ``id["..."]`` label."""
    cleaned = _MERMAUD_ESCAPE_RE.sub(" ", str(text or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_len]


def _node_label(node: Dict[str, Any]) -> str:
    tool = escape_label(str(node.get("tool_name", "tool")), 24)
    summary = escape_label(str(node.get("summary", "")), 62)
    return f"{tool}: {summary}"


def build_mermaid(
    nodes: List[Dict[str, Any]],
    *,
    max_nodes: int = 40,
    max_chars: int = 2000,
    preamble: str = "",
) -> str:
    """Build the Mermaid canvas string from a node list (call order, oldest first).

    Returns a string that fits within ``max_chars``. When there are more than
    ``max_nodes``, the NEWEST nodes are kept (trimmed from the front) and an
    ``OLDER[...]`` stub is appended pointing at ``offload_search`` so the
    model knows more history is retrievable.
    """
    # Input is call order (oldest first). Keep the newest max_nodes.
    trimmed = nodes[-max_nodes:] if len(nodes) > max_nodes else list(nodes)
    ordered = trimmed
    dropped_front = max(0, len(nodes) - max_nodes)

    def render(candidate_nodes: List[Dict[str, Any]], use_subgraphs: bool) -> str:
        lines = ["flowchart LR"]
        if use_subgraphs:
            lines.append("  subgraph offloaded[Offloaded tool calls]")
        for idx, node in enumerate(candidate_nodes):
            node_id = escape_label(str(node.get("node_id", f"N{idx:03d}")), 20)
            label = _node_label(node)
            lines.append(f'  {node_id}["{label}"]')
        # Edges: call order only.
        for i in range(len(candidate_nodes) - 1):
            a = escape_label(str(candidate_nodes[i].get("node_id", "")), 20)
            b = escape_label(str(candidate_nodes[i + 1].get("node_id", "")), 20)
            if a and b:
                lines.append(f"  {a} --> {b}")
        if use_subgraphs:
            lines.append("  end")
        return "\n".join(lines)

    # Ladder: full with subgraphs -> without subgraphs -> halve until fits.
    body = render(ordered, use_subgraphs=True)
    if len(body) > max_chars:
        body = render(ordered, use_subgraphs=False)
    kept = ordered
    dropped_from_budget = 0
    if len(body) > max_chars:
        n = len(ordered)
        while n > 2 and len(body) > max_chars:
            n = max(2, n // 2)
            kept = ordered[-n:]  # newest
            dropped_from_budget = len(ordered) - len(kept)
            body = render(kept, use_subgraphs=False)

    total_dropped = dropped_front + dropped_from_budget
    if total_dropped > 0:
        body = body + (
            f"\n  OLDER[\"...{total_dropped} more offloaded calls — "
            'use offload_search(query=...) to retrieve\"]'
        )

    header = preamble.strip()
    if header:
        return f"{header}\n\n```mermaid\n{body}\n```"
    return f"```mermaid\n{body}\n```"