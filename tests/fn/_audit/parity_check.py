#!/usr/bin/env python3
"""Three-way parity audit: morie Python vs morie/r-package/R vs rmorie/R.

Every function in morie.fn that has an R counterpart must have BOTH R
counterparts, and they must not have drifted. This exists because the
fzcvm p-value bug was present in all three trees and was fixed in only one of
them twice running -- first Python alone, then Python plus rmorie -- because
nobody was checking.

Usage:
    python tests/fn/_audit/parity_check.py            # report
    python tests/fn/_audit/parity_check.py --strict   # exit 1 on any gap
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
PY = REPO / "src" / "morie" / "fn"
R_MORIE = REPO / "r-package" / "morie" / "R"
R_RMORIE = pathlib.Path("/Volumes/VSR/rootcoderfiles/r-morie-oss/R")


# The two packages are legitimately named differently, so a byte comparison
# reports ~88 "differences" that are nothing of the kind: morie:: vs rmorie::,
# package = "morie" vs "rmorie", .morie_sha256_impl vs .rmorie_sha256_impl,
# and RcppExports pointing at different compiled backends. Normalising those
# away is what makes the remaining differences worth looking at.
_RENAMES = (
    (re.compile(r"\brmorie\b"), "PKG"),
    (re.compile(r"\bmorie\b"), "PKG"),
    (re.compile(r"_rmorie_"), "_PKG_"),
    (re.compile(r"_morie_"), "_PKG_"),
    (re.compile(r"\.rmorie_"), ".PKG_"),
    (re.compile(r"\.morie_"), ".PKG_"),
)

# Generated or inherently package-specific; never expected to match.
_EXEMPT = {"RcppExports", "zzz", "morie-package", "rmorie-package"}


def _normalised(p: pathlib.Path) -> str:
    s = p.read_text(errors="replace")
    for rx, sub in _RENAMES:
        s = rx.sub(sub, s)
    # Collapse whitespace so pure reflowing does not register as drift.
    return "\n".join(line.rstrip() for line in s.splitlines() if line.strip())


def _digest(p: pathlib.Path) -> str:
    return hashlib.sha256(_normalised(p).encode()).hexdigest()[:12]


def audit():
    py_mods = {p.stem for p in PY.glob("*.py") if not p.stem.startswith("_")}
    rm = {p.stem for p in R_MORIE.glob("*.R")} if R_MORIE.exists() else set()
    rr = {p.stem for p in R_RMORIE.glob("*.R")} if R_RMORIE.exists() else set()

    both = sorted(py_mods & rm & rr)
    only_rmorie = sorted((py_mods & rr) - rm)
    only_morie_r = sorted((py_mods & rm) - rr)

    drifted = []
    for m in both:
        if m in _EXEMPT:
            continue
        a, b = R_MORIE / f"{m}.R", R_RMORIE / f"{m}.R"
        if _digest(a) != _digest(b):
            drifted.append(m)

    print(f"morie.fn Python modules      : {len(py_mods)}")
    print(f"with BOTH R counterparts     : {len(both)}")
    print(f"R twins that DIFFER          : {len(drifted)}")
    if drifted:
        print("  " + " ".join(drifted[:40]))
    print(f"in rmorie only (no morie/R)  : {len(only_rmorie)}")
    if only_rmorie:
        print("  " + " ".join(only_rmorie[:40]))
    print(f"in morie/R only (no rmorie)  : {len(only_morie_r)}")
    if only_morie_r:
        print("  " + " ".join(only_morie_r[:40]))
    return drifted, only_rmorie, only_morie_r


# A roxygen block that says "Wraps \pkg{Foo}" is only telling the truth if the
# module actually calls Foo. Both packages went native under prose that never
# got updated -- irm.R advertised dispatching to DoubleML directly above a body
# that calls .morie_dml_irm_native() and nothing else.
_CLAIM = re.compile(r"#'.*\b(?:[Ww]raps|[Tt]hin (?:R )?wrapper|dispatches to)\b.*")
_PKGREF = re.compile(r"\\pkg\{([\w.]+)\}|`([\w.]+)`")


def stale_wrapper_claims(tree: pathlib.Path):
    """Roxygen claiming delegation to a package the module never calls."""
    out = []
    for p in sorted(tree.glob("*.R")):
        text = p.read_text(errors="replace")
        code = "\n".join(l for l in text.splitlines()
                         if not l.strip().startswith("#"))
        for lineno, line in enumerate(text.splitlines(), 1):
            if not _CLAIM.search(line):
                continue
            named = {g for t in _PKGREF.findall(line) for g in t if g}
            live = {n for n in named
                    if re.search(rf'["\']{re.escape(n)}["\']|{re.escape(n)}::', code)}
            if named and not live:
                out.append((p.name, lineno, sorted(named), line.strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the two R trees have drifted")
    ap.add_argument("--wrappers", action="store_true",
                    help="report roxygen claiming delegation the code cannot do")
    args = ap.parse_args()

    if args.wrappers:
        bad = 0
        for label, tree in (("morie/R", R_MORIE), ("rmorie/R", R_RMORIE)):
            if not tree.exists():
                continue
            hits = stale_wrapper_claims(tree)
            print(f"\n=== {label}: {len(hits)} stale wrapper claim(s) ===")
            for fname, lineno, named, line in hits:
                print(f"  {fname}:{lineno} [{','.join(named)}]\n      {line[:96]}")
            bad += len(hits)
        # Endpoint/C-function wraps ("the getCubeMetadata POST endpoint",
        # "libsodium's randombytes_buf") are correct and will show up here.
        print("\nReview each: only R-package claims are bugs.")
        return 1 if (args.strict and bad) else 0

    if not R_RMORIE.exists():
        print(f"rmorie checkout not found at {R_RMORIE}; cannot audit", file=sys.stderr)
        return 0
    drifted, _, _ = audit()
    if args.strict and drifted:
        print(f"\nFAIL: {len(drifted)} module(s) differ between the two R trees.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
