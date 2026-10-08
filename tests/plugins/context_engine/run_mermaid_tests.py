"""Ad-hoc runner that executes the mermaid_offload test functions without pytest.

Provides lightweight stand-ins for the pytest fixtures used by the test module
(``tmp_path``, ``monkeypatch``, ``_isolated_home``, ``engine``) so the same
assertions run against the REAL host imports. Returns non-zero on any failure.

Usage:
    ./venv/bin/python3 tests/plugins/context_engine/run_mermaid_tests.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import traceback


class _Monkeypatch:
    def setenv(self, key, value):
        os.environ[key] = value

    def setattr(self, obj, name, value):
        setattr(obj, name, value)

    def delattr(self, obj, name):
        delattr(obj, name)


def _make_fixtures():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="hermes_mermaid_run_"))
    monkey = _Monkeypatch()
    monkey.setenv("HERMES_HOME", str(tmp))

    def _isolated_home(monkeypatch):
        t = pathlib.Path(tempfile.mkdtemp(prefix="hermes_mermaid_test_"))
        monkeypatch.setenv("HERMES_HOME", str(t))
        return t

    def _tmp_path():
        return pathlib.Path(tempfile.mkdtemp(prefix="mermaid_tmppath_"))

    def _engine():
        from plugins.context_engine.mermaid_offload.engine import MermaidOffloadEngine

        eng = MermaidOffloadEngine()
        eng._data_dir = pathlib.Path(tempfile.mkdtemp(prefix="mermaid_store_"))
        return eng

    return {
        "monkeypatch": monkey,
        "_isolated_home": _isolated_home,
        "tmp_path": _tmp_path,
        "engine": _engine,
    }


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    import importlib.util

    test_file = pathlib.Path(__file__).resolve().with_name("test_mermaid_offload.py")
    spec = importlib.util.spec_from_file_location(
        "mermaid_offload_test_module", str(test_file)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fixtures = _make_fixtures()

    passed = 0
    failed = []
    test_names = sorted(n for n in dir(mod) if n.startswith("test_"))
    for name in test_names:
        fn = getattr(mod, name)
        args = []
        import inspect

        sig = inspect.signature(fn)
        missing = []
        for p in sig.parameters:
            if p in fixtures:
                args.append(fixtures[p]())
            elif p in ("tmp_path", "monkeypatch"):
                args.append(fixtures[p]())
            else:
                missing.append(p)
        if missing:
            failed.append((name, f"unhandled params {missing}"))
            continue
        try:
            fn(*args)
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("-" * 60)
    print(f"passed: {passed}, failed: {len(failed)}")
    if failed:
        for name, err in failed:
            print(f"  FAILED {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())