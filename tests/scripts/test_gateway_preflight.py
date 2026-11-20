"""Tests for scripts/gateway_guard/preflight.py (2026-11-18 crash-loop guard)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "gateway_preflight", ROOT / "scripts/gateway_guard/preflight.py"
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

GOOD = (
    "from dataclasses import dataclass\n"
    "from pathlib import Path\n\n"
    "@dataclass(frozen=True)\n"
    "class _CronStorePaths:\n    a: Path\n\n"
    "X = _CronStorePaths(Path('.'))\n"
)
BAD = GOOD.replace("@dataclass(frozen=True)\n", "")  # the 2026-11-18 edit
MODS = ("cron.jobs",)


def git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "cron").mkdir()
    (tmp_path / "cron/__init__.py").write_text("")
    (tmp_path / "cron/jobs.py").write_text(GOOD)
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return tmp_path


def test_clean_tree_passes(repo):
    assert pf.run_preflight(str(repo), sys.executable, MODS) == []
    assert pf.guard(str(repo), sys.executable, modules=MODS) == 0


def test_missing_decorator_is_caught_and_names_file(repo):
    (repo / "cron/jobs.py").write_text(BAD)
    fails = pf.run_preflight(str(repo), sys.executable, MODS)
    assert len(fails) == 1
    mod, where, err = fails[0]
    assert mod == "cron.jobs" and where.startswith("cron/jobs.py:")
    assert "TypeError" in err
    assert "cron/jobs.py" in pf.format_failures(fails)


def test_guard_stashes_dirty_tree_and_starts(repo, tmp_path):
    (repo / "cron/jobs.py").write_text(BAD)
    log = tmp_path / "pf.log"
    assert pf.guard(str(repo), sys.executable, str(log), MODS) == 0
    assert (repo / "cron/jobs.py").read_text() == GOOD
    stash = pf._run(
        ["git", "-C", str(repo), "stash", "list"],
        capture_output=True,
        text=True,
    ).stdout
    assert "auto-stash: gateway preflight failed" in stash
    assert "cron/jobs.py" in log.read_text()


def test_guard_refuses_and_restores_when_stash_is_not_the_cause(repo):
    (repo / "cron/jobs.py").write_text(GOOD + "\nimport not_a_real_module_xyz\n")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "broken")
    (repo / "cron/__init__.py").write_text("# harmless uncommitted edit\n")
    assert pf.guard(str(repo), sys.executable, modules=MODS) == 1
    assert (repo / "cron/__init__.py").read_text().startswith("# harmless")  # restored
    stash = pf._run(
        ["git", "-C", str(repo), "stash", "list"],
        capture_output=True,
        text=True,
    ).stdout
    assert stash == ""


def test_clean_but_broken_tree_refuses(repo):
    (repo / "cron/jobs.py").write_text(BAD)
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "broken")
    assert pf.guard(str(repo), sys.executable, modules=MODS) == 1


def test_real_cron_jobs_without_decorator_is_caught(tmp_path):
    src = (ROOT / "cron/jobs.py").read_text()
    assert "@dataclass(frozen=True)\nclass _CronStorePaths" in src
    (tmp_path / "cron").mkdir()
    (tmp_path / "cron/__init__.py").write_text("")
    (tmp_path / "cron/jobs.py").write_text(src)
    if pf.run_preflight(str(tmp_path), sys.executable, MODS):
        pytest.skip("real cron.jobs not importable from a bare copy here")
    (tmp_path / "cron/jobs.py").write_text(
        src.replace(
            "@dataclass(frozen=True)\nclass _CronStorePaths", "class _CronStorePaths"
        )
    )
    fails = pf.run_preflight(str(tmp_path), sys.executable, MODS)
    assert fails and "takes no arguments" in fails[0][2]
    assert fails[0][1].startswith("cron/jobs.py:")


def test_real_checkout_passes_preflight():
    assert pf.run_preflight(str(ROOT), sys.executable) == []
