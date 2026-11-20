#!/usr/bin/env python3
"""Gateway start preflight + auto-stash guard (stdlib only, standalone).

Run before every gateway start/restart (launchd wrapper). Imports the gateway
entrypoint modules in a throwaway subprocess so a broken edit fails here, loudly,
instead of crash-looping the daemon (incident 2026-11-18).

Exit codes: 0 start the gateway (clean, or recovered by stashing);
            1 do NOT start (broken and stash did not help / nothing to stash);
            2 guard-internal error (wrapper should fail open and start anyway).
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys

MODULES = (
    "hermes_cli.main",
    "hermes_cli.gateway",
    "gateway.run",
    "gateway.platforms.api_server",
    "cron.jobs",
)

# Child: import each module, report every failure with the innermost frame in the repo.
_CHILD = r"""
import importlib, os, sys, traceback
repo = os.path.realpath(os.getcwd())
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except BaseException as e:
        where = "?"
        for fs in reversed(traceback.extract_tb(e.__traceback__)):
            if os.path.realpath(fs.filename).startswith(repo + os.sep):
                where = "%s:%s" % (os.path.relpath(os.path.realpath(fs.filename), repo), fs.lineno)
                break
        if isinstance(e, SyntaxError) and e.filename:
            where = "%s:%s" % (os.path.relpath(os.path.realpath(e.filename), repo), e.lineno)
        print("PREFLIGHT_FAIL\t%s\t%s\t%s: %s" % (name, where, type(e).__name__, e))
"""


def _run(*a, **kw):
    return subprocess.run(*a, check=False, **kw)


def run_preflight(repo: str, python: str, modules=MODULES, timeout: int = 120):
    """Return a list of (module, file:line, error) failures; empty means OK."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        p = _run(
            [python, "-c", _CHILD, *modules],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [("(all)", "?", f"import timed out after {timeout}s")]
    fails = []
    for line in p.stdout.splitlines():
        if line.startswith("PREFLIGHT_FAIL\t"):
            _, mod, where, err = line.split("\t", 3)
            fails.append((mod, where, err))
    if p.returncode != 0 and not fails:  # child died (segfault, interpreter missing...)
        fails.append(
            ("(all)", "?", f"child exit {p.returncode}: {p.stderr.strip()[-300:]}")
        )
    return fails


def format_failures(fails) -> str:
    return "\n".join(
        f"PREFLIGHT FAILED: {w}  (importing {m}) -> {e}" for m, w, e in fails
    )


def _git(repo, *args):
    return _run(["git", "-C", repo, *args], capture_output=True, text=True)


def _alert(msg: str, log_path: str | None) -> None:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(msg, file=sys.stderr, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"{stamp} {msg}\n")
    cmd = os.environ.get("HERMES_PREFLIGHT_ALERT_CMD")  # optional: message on stdin
    if cmd:
        _run(cmd, shell=True, input=msg, text=True, timeout=30)


def guard(repo: str, python: str, log_path: str | None = None, modules=MODULES) -> int:
    fails = run_preflight(repo, python, modules)
    if not fails:
        return 0
    msg = format_failures(fails)
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if not dirty:
        _alert(
            msg
            + "\nTree is clean (committed code is broken); nothing to stash. NOT starting.",
            log_path,
        )
        return 1
    stamp = datetime.datetime.now().strftime("%F_%T")
    label = f"auto-stash: gateway preflight failed {stamp}"
    r = _git(
        repo, "stash", "push", "-m", label
    )  # tracked files only; untracked left alone
    if r.returncode != 0:
        _alert(msg + f"\nstash failed: {r.stderr.strip()}. NOT starting.", log_path)
        return 1
    if run_preflight(repo, python, modules):
        pop = _git(repo, "stash", "pop")  # stash was not the cause: put the work back
        _alert(
            msg + "\nStashing did not fix it (still broken on clean tree); "
            f"changes restored (pop rc={pop.returncode}). NOT starting.",
            log_path,
        )
        return 1
    _alert(
        msg
        + f"\nCAUSE: uncommitted edits. Stashed as '{label}' (see `git stash list`); "
        "starting from the clean tree. Recover with `git stash pop`.",
        log_path,
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log", help="append alerts here")
    ap.add_argument("--check-only", action="store_true", help="no stash, just report")
    a = ap.parse_args(argv)
    try:
        if a.check_only:
            fails = run_preflight(a.repo, a.python)
            if fails:
                print(format_failures(fails), file=sys.stderr)
            return 1 if fails else 0
        return guard(a.repo, a.python, a.log)
    except Exception as e:  # guard bug must not take the gateway down
        print(f"preflight guard internal error: {e!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
