#!/usr/bin/env python3
"""Triage the auto-extracted placeholder modules in morie.fn.

Auto-extracted modules are named <book><chapter>u<equation>.py and
carry a RichResult title ending "(auto-extracted; see ref)". They are
machine-harvested one-per-equation stubs; this script classifies each
so a keep/delete decision can be made per module rather than per pile.

Writes results INCREMENTALLY (flushed every row) to the CSV so a
killed or interrupted run still leaves everything it had reached.
Resumable: re-running skips modules already present in the CSV.

Verdicts
--------
implementable   the docstring states a formula AND the signature has
                data-shaped parameters -- a real target
formula-only    a formula is stated but the signature is degenerate
                (single unnamed x, or no params) -- would need the
                book to reconstruct the call contract
duplicate       formula text identical to an earlier module (the
                harvester emitted the same equation many times)
no-formula      no formula line at all -- nothing to implement from
prose-artifact  the "formula" is prose or a fragment, not an equation
                (no operator, no LaTeX, under 12 chars)
"""

import ast
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

FN_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "src/morie/fn")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "scripts/audit/autoextracted_triage.csv")

AUTO_RE = re.compile(r"^(?P<book>.+?)(?P<ch>\d+)u(?P<eq>\d+)\.py$")
FORMULA_RE = re.compile(r"^\s*Formula:\s*(.+)$", re.MULTILINE)
OPERATORS = set("=+-*/^<>∑∫√±≤≥≠αβγδθλμσπΦφ")

FIELDS = [
    "module", "book", "chapter", "equation", "function", "n_params",
    "params", "formula", "formula_hash", "verdict", "reason",
]


def is_placeholder(text):
    return "result = float(np.mean(" in text and "se = float(np.std(" in text


def classify(formula, params):
    if not formula:
        return "no-formula", "no Formula: line in the docstring"
    f = formula.strip()
    if len(f) < 12 and not any(c in OPERATORS for c in f):
        return "prose-artifact", f"formula too short and operator-free: {f!r}"
    if not any(c in OPERATORS for c in f) and "\\" not in f:
        return "prose-artifact", "no operator or LaTeX in the formula text"
    data_params = [p for p in params if p not in ("self", "cdf", "method", "kwargs")]
    if not data_params:
        return "formula-only", "no data parameters in the signature"
    if len(data_params) == 1 and data_params[0] in ("x", "data", "y"):
        return "formula-only", "single generic parameter; call contract unclear"
    return "implementable", f"formula plus {len(data_params)} data parameters"


def main():
    done = set()
    if OUT.exists():
        with OUT.open(newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(row["module"])
        print(f"resuming: {len(done)} modules already triaged", flush=True)

    seen_hashes = {}
    if done:
        with OUT.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["formula_hash"]:
                    seen_hashes.setdefault(row["formula_hash"], row["module"])

    files = sorted(p for p in FN_DIR.glob("*.py") if AUTO_RE.match(p.name))
    print(f"{len(files)} auto-extracted candidates on disk", flush=True)

    new = OUT.exists()
    with OUT.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not new:
            w.writeheader()
            fh.flush()
        n = 0
        for p in files:
            if p.name in done:
                continue
            m = AUTO_RE.match(p.name)
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                w.writerow({
                    "module": p.name, "book": m["book"], "chapter": m["ch"],
                    "equation": m["eq"], "function": "", "n_params": "",
                    "params": "", "formula": "", "formula_hash": "",
                    "verdict": "unreadable", "reason": str(exc),
                })
                fh.flush()
                continue
            if not is_placeholder(text):
                continue

            fn_name, params = "", []
            try:
                tree = ast.parse(text)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name != "cheatsheet":
                        fn_name = node.name
                        params = [a.arg for a in node.args.args]
                        break
            except SyntaxError:
                pass

            fm = FORMULA_RE.search(text)
            formula = fm.group(1).strip() if fm else ""
            fhash = (
                hashlib.sha1(re.sub(r"\s+", " ", formula).encode()).hexdigest()[:12]
                if formula else ""
            )

            verdict, reason = classify(formula, params)
            if verdict == "implementable" and fhash in seen_hashes:
                verdict = "duplicate"
                reason = f"same formula as {seen_hashes[fhash]}"
            elif fhash and fhash not in seen_hashes:
                seen_hashes[fhash] = p.name

            w.writerow({
                "module": p.name, "book": m["book"], "chapter": m["ch"],
                "equation": m["eq"], "function": fn_name,
                "n_params": len(params), "params": "|".join(params),
                "formula": formula[:300], "formula_hash": fhash,
                "verdict": verdict, "reason": reason,
            })
            fh.flush()
            os.fsync(fh.fileno())
            n += 1
            if n % 500 == 0:
                print(f"  {n} triaged", flush=True)
    print(f"DONE: {n} newly triaged, CSV at {OUT}", flush=True)


if __name__ == "__main__":
    main()
