#!/usr/bin/env python3
"""Harvest every test the FALSE-POSITIVE RISK detector fires on, into a CSV.

The detector lives in ``tests/conftest.py`` as an autouse fixture. It is a
pure source-inspection check -- it reads ``inspect.getsource(request.function)``
and looks at the assert lines -- so it can be replayed statically over the
whole tree without running pytest. That matters: the full ``tests/fn`` run is
11+ minutes on six workers, and the warning it emits does not fail a build,
so nothing has ever read it.

This reproduces the fixture's logic exactly (same pattern set, same
``startswith`` prefix match on the stripped line, same "all asserts trivial
and at least one assert" condition) and adds the context needed to act on a
row: which function is under test, whether that target has a real body or a
shared stub skeleton, and what the assert lines actually say.

Usage::

    python tests/fn/_audit/harvest_trivial.py            # writes trivial_tests.csv
    python tests/fn/_audit/harvest_trivial.py --check    # exit 1 if count grew
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TESTS = REPO / "tests"
FN_SRC = REPO / "src" / "morie" / "fn"
OUT = Path(__file__).resolve().parent / "trivial_tests.csv"

# A skeleton shared by at least this many functions is a generated stub body,
# not an implementation. The threshold is the audit README's; the top two
# skeletons alone cover 64% of the tree.
SHARED_SKELETON_MIN = 20

# Copied verbatim from tests/conftest.py::_TRIVIAL_PATTERNS. If that set is
# edited, edit this one in the same commit -- test_audit_trivial_harvest.py
# asserts the two are identical, so a drift fails the build rather than
# silently changing what the harvest counts.
TRIVIAL_PATTERNS = {
    "assert True",
    "assert result is not None",
    "assert r.value is not None",
    "assert r is not None",
    "assert result is result",
    "assert r is r",
    "assert r.name",
    "assert result is not None or",
    "assert r.value is not None or",
}


def _assert_lines(src: str) -> list[str]:
    """The fixture's own line selection: stripped lines starting with `assert`."""
    return [ln.strip() for ln in src.splitlines() if ln.strip().startswith("assert")]


def _is_trivial(lines: list[str]) -> bool:
    """The fixture's condition: at least one assert, and every one is trivial."""
    if not lines:
        return False
    trivial = sum(1 for ln in lines if any(ln.startswith(p) for p in TRIVIAL_PATTERNS))
    return trivial == len(lines)


def _target_of(tree: ast.Module) -> str:
    """Best-effort name of the function under test, from the module's imports.

    Generated tests import exactly one name from ``morie.fn.<mod>``, so the
    first such import is the target. Falls back to empty when the module
    imports nothing from morie.fn (a hand-written test may not).
    """
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("morie.fn."):
            if node.names:
                return node.names[0].name
    return ""


def _module_of(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("morie.fn."):
            return (node.module or "").rsplit(".", 1)[-1]
    return ""


class _Skeletonise(ast.NodeTransformer):
    """Erase identifiers and literals, keeping only control/call structure."""

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        return ast.copy_location(ast.Attribute(value=node.value, attr="_", ctx=node.ctx), node)

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=0), node)

    def visit_arg(self, node):
        return ast.copy_location(ast.arg(arg="_", annotation=None), node)


def _skeletons() -> tuple[dict[tuple[str, str], str], collections.Counter]:
    """Map (module, function) -> body skeleton, plus how many share each one.

    ``cheatsheet`` is excluded: every module has one and every one is a bare
    ``return "..."``, so including them would put 36k identical bodies in the
    census and say nothing about whether the function under test is real.
    """
    per_fn: dict[tuple[str, str], str] = {}
    census: collections.Counter = collections.Counter()
    for path in sorted(FN_SRC.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == "cheatsheet":
                continue
            # Drop the docstring: two functions differing only in prose are
            # the same body.
            body = [
                s
                for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            ]
            if not body:
                continue
            module = ast.Module(body=body, type_ignores=[])
            # Round-trip through unparse so equivalent trees normalise.
            skel = ast.dump(_Skeletonise().visit(ast.parse(ast.unparse(module))))
            per_fn[(path.stem, node.name)] = skel
            census[skel] += 1
    return per_fn, census


def harvest() -> list[dict]:
    rows: list[dict] = []
    per_fn, census = _skeletons()
    for path in sorted(TESTS.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        target = _target_of(tree)
        module = _module_of(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            # ast line numbers are 1-based and exclude decorators, which is
            # what inspect.getsource includes -- but decorator lines never
            # start with `assert`, so the assert selection is unaffected.
            src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            asserts = _assert_lines(src)
            if not _is_trivial(asserts):
                continue
            skel = per_fn.get((module, target))
            siblings = census[skel] if skel is not None else 0
            if skel is None:
                target_kind = "unknown"
            elif siblings >= SHARED_SKELETON_MIN:
                target_kind = "stub"
            else:
                target_kind = "real"
            rows.append(
                {
                    "nodeid": f"{path.relative_to(REPO)}::{node.name}",
                    "test_file": str(path.relative_to(REPO)),
                    "test_name": node.name,
                    "module": module,
                    "target_function": target,
                    "target_kind": target_kind,
                    "skeleton_siblings": siblings,
                    "n_asserts": len(asserts),
                    "assert_lines": " | ".join(asserts),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed CSV and exit 1 if the count grew",
    )
    args = ap.parse_args()

    rows = harvest()

    if args.check:
        if not OUT.exists():
            print(f"{OUT} missing; run without --check first", file=sys.stderr)
            return 1
        with OUT.open(newline="", encoding="utf-8") as fh:
            committed = sum(1 for _ in csv.DictReader(fh))
        print(f"committed={committed} current={len(rows)}")
        if len(rows) > committed:
            print(
                f"FAIL: {len(rows) - committed} new trivial test(s) added. "
                "Tighten the assertions or update the baseline deliberately.",
                file=sys.stderr,
            )
            return 1
        return 0

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "nodeid",
                "test_file",
                "test_name",
                "module",
                "target_function",
                "target_kind",
                "skeleton_siblings",
                "n_asserts",
                "assert_lines",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = collections.Counter(r["target_kind"] for r in rows)
    print(f"{len(rows)} trivial tests -> {OUT.relative_to(REPO)}")
    print(
        f"  target real={counts['real']}  stub={counts['stub']}  "
        f"unknown={counts['unknown']}"
    )
    print("  'real' rows are the actionable ones; 'stub' rows are blocked on")
    print("  the stub itself and are out of scope for the red series.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
