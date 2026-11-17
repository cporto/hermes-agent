"""Shared file-serving allowlist primitives (single source of truth).

Owns the roots, deny-list, and the pure path validator used by BOTH the streaming
GET /v1/files download and the batch POST /v1/files/validate confirmation oracle, so
"validate says ok" and "download returns 200" can never drift apart. This module is
deliberately INERT: no request handling, no streaming, no response building — only the
pure validation / deny-list / roots helpers re-exported back into ``api_server``.

Every check runs against the canonical (symlink-resolved) path. The raw input is only
ever echoed back to the client; the resolved/canonical path is never sent in a response.
"""

import fnmatch
import os
import stat
from typing import Optional

# Absolute allowlist roots, resolved once from the runtime $HOME (never hardcoded
# /home/... guesses). The canonical path stays in the module so both consumers share it.
_FILES_MAX_BYTES = 50 * 1024 * 1024


def _files_roots() -> list[str]:
    """Absolute allowlist roots, resolved once from the runtime $HOME."""
    home = os.path.expanduser("~")
    raw = [
        os.path.join(home, "Projects"),
        os.path.join(home, "workspace"),
        os.path.join(home, ".hermes"),
        os.path.join(home, ".hermes", "cache", "scratch"),
        os.path.join(home, "Downloads"),
        "/tmp",
    ]
    return [os.path.realpath(r) for r in raw]


def _files_deny_dirs() -> list[str]:
    """Absolute directories that are never served even though they sit under an allowed root."""
    home = os.path.expanduser("~")
    gateway_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    return [
        os.path.realpath(os.path.join(home, ".ssh")),
        os.path.realpath(os.path.join(home, ".hermes", "keys")),
        gateway_dir,
    ]


# Any path segment matching one of these (e.g. `.git`) denies the whole path.
_FILES_DENY_SEGMENTS = (".git",)
# Leaf-name patterns that are never served even inside an allowed root.
_FILES_DENY_NAMES = (
    ".env", ".env.*", "*.pem", "*.key", "id_rsa", "id_ed25519",
    "known_hosts", "authorized_keys", "API_SERVER_KEY", "state.db",
    "*.sqlite*", "package-lock.json", "config.yaml", "config.yml",
)


def _validate_allowlist_path(raw_path: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(canonical_allowed_path, None)`` if safe to serve, else ``(None, deny_reason)``."""
    if not raw_path:
        return None, "missing_path"
    if "\x00" in raw_path or any(ord(c) < 32 for c in raw_path):
        return None, "bad_path"
    if "\\" in raw_path:
        return None, "backslash"
    expanded = os.path.expanduser(raw_path)
    if not os.path.isabs(expanded):
        return None, "not_absolute"
    if any(part == ".." for part in expanded.replace("\\", "/").split("/")):
        return None, "traversal"
    canonical = os.path.realpath(expanded)
    for deny_dir in _files_deny_dirs():
        if canonical == deny_dir or canonical.startswith(deny_dir + os.sep):
            return None, "denied"
    allowed = False
    for root in _files_roots():
        if canonical == root or canonical.startswith(root + os.sep):
            allowed = True
            break
    if not allowed:
        return None, "not_allowed"
    parts = canonical.split(os.sep)
    if any(seg in _FILES_DENY_SEGMENTS for seg in parts):
        return None, "denied"
    leaf = parts[-1] if parts else canonical
    if any(fnmatch.fnmatch(leaf, pat) for pat in _FILES_DENY_NAMES):
        return None, "denied"
    try:
        st = os.stat(canonical)
    except FileNotFoundError:
        return None, "not_found"
    except OSError:
        return None, "read_error"
    if not stat.S_ISREG(st.st_mode):
        return None, "not_a_file"
    if st.st_nlink != 1:
        return None, "hardlink"
    if st.st_size > _FILES_MAX_BYTES:
        return None, "too_large"
    return canonical, None
