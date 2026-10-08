"""Disk-backed storage for offloaded tool output.

Layout (per session)::

    <data_dir>/<session_id>/
        index.jsonl          # append-only node index: one JSON object per line
        refs/N001.md         # full raw tool output for node N001
        canvas.mmd           # generated Mermaid flowchart for the session

Design notes (see the reference implementation):
  * Append-only JSONL: a crash mid-compression cannot corrupt prior entries.
  * Cache entries in memory; reload on ``on_session_start`` and continue node
    numbering from ``max(node_id, seq) + 1`` so a resumed session doesn't
    collide.
  * Skip corrupt index lines individually rather than failing the whole load.
  * Sanitize ``session_id`` before using it as a path component — it comes
    from the host.
  * **Degrade, never raise.** If ``mkdir`` fails, mark the store degraded, log
    once, and leave tool output untouched. If a ref write fails, do NOT write
    the placeholder — never reference content that isn't on disk.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SESSION_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_NODE_ID_RE = re.compile(r"[nN]?0*(\d+)")


def sanitize_session_id(session_id: str) -> str:
    """Sanitize a host-provided session id for safe use as a path component."""
    cleaned = _SESSION_ID_SAFE_RE.sub("_", str(session_id or "default"))
    cleaned = cleaned.strip("._")
    return (cleaned[:128] or "default")


def parse_node_id(node_id: Any) -> Optional[int]:
    """Parse a sloppy node id into its integer sequence number.

    Accepts ``N001``, ``n1``, ``1``, ``" N012 "``, ``N001``, etc.
    Returns ``None`` if it cannot be parsed.
    """
    text = str(node_id or "").strip()
    m = _NODE_ID_RE.fullmatch(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def format_node_id(seq: int) -> str:
    """Canonical node id, e.g. ``N001``."""
    return f"N{seq:03d}"


class OffloadStore:
    """Per-session append-only store for offloaded tool output.

    The store is intentionally safe to construct with no I/O. All disk work
    happens in ``open_for_session()`` / ``ensure_open()`` so discovery
    instantiating the engine never touches the filesystem.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._lock = threading.RLock()
        self._session_dir: Optional[Path] = None
        self._index_path: Optional[Path] = None
        self._refs_dir: Optional[Path] = None
        self._canvas_path: Optional[Path] = None
        self._index_fh = None
        self._nodes: Dict[int, Dict[str, Any]] = {}  # seq -> node
        self._seq = 0
        self.degraded = False
        self._degraded_reason = ""
        self._degraded_logged = False
        self.session_id: Optional[str] = None

    # -- open / close -----------------------------------------------------

    def open_for_session(self, session_id: str) -> bool:
        """Open (or reload) storage for a session. Returns True if usable."""
        sid = sanitize_session_id(session_id)
        self.session_id = sid
        with self._lock:
            try:
                self._session_dir = self.data_dir / sid
                self._session_dir.mkdir(parents=True, exist_ok=True)
                self._refs_dir = self._session_dir / "refs"
                self._refs_dir.mkdir(parents=True, exist_ok=True)
                self._index_path = self._session_dir / "index.jsonl"
                self._canvas_path = self._session_dir / "canvas.mmd"
            except OSError as exc:  # degrade, never raise
                self._mark_degraded(f"cannot create session dir: {exc}")
                return False

            self._load_index()
            return not self.degraded

    def close(self) -> None:
        with self._lock:
            if self._index_fh is not None:
                try:
                    self._index_fh.close()
                except OSError:
                    pass
                self._index_fh = None

    def _mark_degraded(self, reason: str) -> None:
        self.degraded = True
        self._degraded_reason = reason
        if not self._degraded_logged:
            logger.warning("mermaid_offload store degraded: %s", reason)
            self._degraded_logged = True

    def _load_index(self) -> None:
        """Load existing index entries; skip corrupt lines individually."""
        assert self._index_path is not None
        max_seq = 0
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # skip corrupt line, keep going
                        seq = entry.get("seq")
                        if isinstance(seq, int):
                            self._nodes[seq] = entry
                            max_seq = max(max_seq, seq)
            except OSError as exc:
                self._mark_degraded(f"cannot read index: {exc}")
                return
        self._seq = max_seq
        # Reopen index for append.
        try:
            self._index_fh = open(self._index_path, "a", encoding="utf-8")
        except OSError as exc:
            self._mark_degraded(f"cannot open index for append: {exc}")

    # -- mutation ---------------------------------------------------------

    def add(
        self,
        *,
        tool_name: str,
        summary: str,
        content: str,
        placeholder_hint: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append a node and write its ref file.

        Returns the node dict on success, or ``None`` if the write failed
        (store degraded) — callers MUST NOT point a placeholder at content
        that isn't on disk.
        """
        with self._lock:
            if self.degraded or self._session_dir is None:
                return None
            self._seq += 1
            seq = self._seq
            node_id = format_node_id(seq)
            now = time.time()
            entry = {
                "seq": seq,
                "node_id": node_id,
                "tool_name": tool_name,
                "summary": summary,
                "placeholder_hint": placeholder_hint,
                "ts": now,
                "ref": f"refs/{node_id}.md",
                "metadata": metadata or {},
            }
            # Write ref file FIRST; only record in index if the ref landed.
            assert self._refs_dir is not None, "store not opened for a session"
            ref_path = self._refs_dir / f"{node_id}.md"
            try:
                ref_path.write_text(content, encoding="utf-8")
            except OSError as exc:
                self._mark_degraded(f"ref write failed: {exc}")
                return None
            try:
                assert self._index_fh is not None
                self._index_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._index_fh.flush()
            except OSError as exc:
                self._mark_degraded(f"index append failed: {exc}")
                return None
            self._nodes[seq] = entry
            return entry

    def get(self, seq: int) -> Optional[Dict[str, Any]]:
        """Return the node entry for a sequence number (index only)."""
        with self._lock:
            return self._nodes.get(seq)

    def read_content(self, node_id: str) -> Optional[str]:
        """Return the full raw content for a node id, or None on miss/error."""
        seq = parse_node_id(node_id)
        if seq is None:
            return None
        with self._lock:
            entry = self._nodes.get(seq)
            if entry is None or self._session_dir is None:
                return None
            ref = entry.get("ref")
            if not ref:
                return None
            ref_path = self._session_dir / ref
            try:
                return ref_path.read_text(encoding="utf-8")
            except OSError:
                return None

    def read_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return a node dict with the full content attached (``content`` key)."""
        seq = parse_node_id(node_id)
        if seq is None:
            return None
        with self._lock:
            entry = self._nodes.get(seq)
            if entry is None:
                return None
            content = self.read_content(entry["node_id"])
            if content is None:
                return None
            out = dict(entry)
            out["content"] = content
            return out

    def search(self, query: str, limit: int = 8, max_chars: int = 10000) -> List[Dict[str, Any]]:
        """Search nodes by summary/tool-name first, then ref bodies.

        Returns node dicts with ``content`` attached, capped at ``max_chars``
        each (with a truncation note), newest-first within each tier.
        """
        q = (query or "").strip().lower()
        if not q:
            return []
        selected_seqs: List[int] = []
        with self._lock:
            # Tier 1: summary / tool_name / metadata matches, newest-first.
            for seq in sorted(self._nodes, reverse=True):
                entry = self._nodes[seq]
                haystack = " ".join(
                    [
                        entry.get("tool_name", ""),
                        entry.get("summary", ""),
                        str(entry.get("placeholder_hint", "")),
                        json.dumps(entry.get("metadata", {}), ensure_ascii=False),
                    ]
                ).lower()
                if q in haystack:
                    selected_seqs.append(seq)
            # Tier 2: scan ref bodies newest-first until the limit fills.
            if len(selected_seqs) < limit:
                for seq in sorted(self._nodes, reverse=True):
                    if seq in selected_seqs:
                        continue
                    entry = self._nodes[seq]
                    content = self.read_content(entry["node_id"])
                    if content and q in content.lower():
                        selected_seqs.append(seq)
                        if len(selected_seqs) >= limit:
                            break

        results: List[Dict[str, Any]] = []
        for seq in selected_seqs[:limit]:
            entry = self._nodes[seq]
            content = self.read_content(entry["node_id"]) or ""
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n...[truncated; full ref: {entry.get('ref')}]..."
            out = dict(entry)
            out["content"] = content
            results.append(out)
        return results

    def available_nodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` node entries (index only)."""
        with self._lock:
            seqs = sorted(self._nodes, reverse=True)[:limit]
            return [dict(self._nodes[s]) for s in seqs]

    def write_canvas(self, text: str) -> bool:
        """Persist the generated Mermaid canvas. Best-effort."""
        if self.degraded or self._canvas_path is None:
            return False
        try:
            self._canvas_path.write_text(text, encoding="utf-8")
            return True
        except OSError:
            return False

    def sweep(self, retention_days: int = 14) -> int:
        """Delete orphaned session dirs older than ``retention_days``.

        Returns number of directories removed. Only touches sibling session
        directories under ``self.data_dir`` (never the current session).
        """
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        current = self._session_dir
        try:
            for child in self.data_dir.iterdir():
                if not child.is_dir() or child.name.startswith((".", "_")):
                    continue
                if current is not None and child.resolve() == current.resolve():
                    continue
                try:
                    if child.stat().st_mtime < cutoff:
                        import shutil

                        shutil.rmtree(child, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
        except OSError:
            pass
        return removed

    @property
    def total_nodes(self) -> int:
        with self._lock:
            return len(self._nodes)

    def __len__(self) -> int:
        return self.total_nodes

    def __deepcopy__(self, memo: Dict[int, Any]) -> "OffloadStore":
        """Deepcopy-safe: file handle and locks are NOT copied.

        A deepcopy of the engine (per child agent) must not share the open
        index file handle or the RLock. Copy only the pure data (nodes,
        seq, paths) and leave the copy closed/degraded-safe (it reopens on
        the next session start).
        """
        new = OffloadStore(self.data_dir)
        new.session_id = self.session_id
        new._session_dir = self._session_dir
        new._refs_dir = self._refs_dir
        new._index_path = self._index_path
        new._canvas_path = self._canvas_path
        new._nodes = dict(self._nodes)
        new._seq = self._seq
        new.degraded = self.degraded
        new._degraded_reason = self._degraded_reason
        # Do NOT copy _index_fh (file handle) — the copy reopens on demand.
        memo[id(self)] = new
        return new