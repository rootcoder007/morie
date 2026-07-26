#!/usr/bin/env python3
"""Local-vs-CI equivalence check for optional-dependency gates.

The problem this exists to solve
--------------------------------
A dev machine that has an optional extra installed and a CI runner that does
not are two different environments, and a test file gated on that extra
reports a *different outcome* in each: PASS locally, SKIP on CI. Both are
green, but they are green for different reasons, and a matrix that records
only one of them is not reproducible on the other machine.

Concretely, on 2026-07-26 the tests/fn re-run showed `test_douml` as 2 PASS
because DoubleML had been installed locally to verify the gate's present
path. CI shows those same two tests as 2 SKIP. Reading the local matrix
alone, you would not know the difference existed.

So: never assert "CI will skip this" from memory. Uninstall the extra, run
the tests, observe the skip, reinstall, run again, observe the pass. Both
paths, measured, every time a gate is added or changed.

Usage
-----
    python tests/fn/_audit/ci_sim.py doubleml tests/fn/test_douml.py
    python tests/fn/_audit/ci_sim.py doubleml tests/fn/test_douml.py tests/fn/test_dml.py

Exit status is 0 only when BOTH paths are clean:
  - extra absent  -> the gated tests SKIP (they must not error)
  - extra present -> the gated tests PASS

Anything else is a broken gate and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# The venv's own interpreter, so this works under uv-created envs that have
# no `pip` of their own (see reference_l14_test_environment).
PY = sys.executable


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _pytest(paths: list[str]) -> tuple[int, int, int, str]:
    """Run pytest, return (passed, skipped, failed, tail-of-output).

    -o addopts="" is mandatory: pyproject.toml excludes tests/fn by default,
    so without it pytest collects nothing and reports a vacuous success.
    """
    proc = _run([PY, "-m", "pytest", *paths, "-q", "-o", "addopts=",
                 "-p", "no:cacheprovider", "-rs"])
    out = proc.stdout + proc.stderr
    def n(word: str) -> int:
        m = re.search(rf"(\d+) {word}", out)
        return int(m.group(1)) if m else 0
    return n("passed"), n("skipped"), n("failed"), out.strip().splitlines()[-1] if out.strip() else ""


def _uninstall(pkg: str) -> None:
    # uv first (the venvs on L14 are uv-created and have no pip); fall back.
    if _run(["uv", "pip", "uninstall", "--python", PY, pkg]).returncode != 0:
        _run([PY, "-m", "pip", "uninstall", "-y", pkg])


def _install(pkg: str) -> None:
    if _run(["uv", "pip", "install", "--quiet", "--python", PY, pkg]).returncode != 0:
        _run([PY, "-m", "pip", "install", "--quiet", pkg])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extra", help="distribution name of the optional dependency, e.g. doubleml")
    ap.add_argument("paths", nargs="+", help="test files or node ids to check")
    ap.add_argument("--json", type=Path, help="write the two-view result here")
    args = ap.parse_args()

    print(f"[ci_sim] extra={args.extra}  paths={' '.join(args.paths)}")

    # --- CI path: the extra is absent -------------------------------------
    _uninstall(args.extra)
    ci_pass, ci_skip, ci_fail, ci_tail = _pytest(args.paths)
    print(f"[ci_sim] CI-equivalent (absent) : {ci_pass} passed, {ci_skip} skipped, {ci_fail} failed")

    # --- local path: the extra is present ---------------------------------
    _install(args.extra)
    lo_pass, lo_skip, lo_fail, lo_tail = _pytest(args.paths)
    print(f"[ci_sim] local       (present) : {lo_pass} passed, {lo_skip} skipped, {lo_fail} failed")

    ok = True
    if ci_fail:
        print(f"[ci_sim] FAIL: a missing optional extra must never fail the suite -- {ci_tail}")
        ok = False
    if ci_skip == 0:
        print("[ci_sim] FAIL: nothing skipped with the extra absent -- the gate is not wired up")
        ok = False
    if lo_fail:
        print(f"[ci_sim] FAIL: gated tests do not pass with the extra installed -- {lo_tail}")
        ok = False
    if lo_skip >= ci_skip and ci_skip:
        print("[ci_sim] FAIL: installing the extra did not un-skip anything")
        ok = False

    if args.json:
        args.json.write_text(json.dumps({
            "extra": args.extra,
            "paths": args.paths,
            "ci_equivalent": {"passed": ci_pass, "skipped": ci_skip, "failed": ci_fail},
            "local":         {"passed": lo_pass, "skipped": lo_skip, "failed": lo_fail},
            "ok": ok,
        }, indent=2) + "\n")

    print(f"[ci_sim] {'OK -- both paths clean' if ok else 'BROKEN GATE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
