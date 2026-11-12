# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sweep every quantile / median / IQR / percentile helper for the infinite-value flaw.

rmoriebricklayer 1.1.1 fixed cqnt(), cmed(), ciqr() and ctkyfnc(): a type-7 quantile written
lo + g * (hi - lo) gives NaN when a neighbour is infinite (0 * Inf, or Inf - Inf), and Inf when
two finite values differ by more than the largest double. This finds every function in
src/morie whose name says quantile, median, percentile or IQR (by AST, not by eye), calls it on
small inputs holding Inf, and compares with R's quantile(type = 7) / median() / IQR(), whose
values are written out below (R 4.6.1, sprintf("%.17g")). Functions it cannot call with
(values) or (values, p) are listed as not probed.

    python -I scripts/infq_sweep.py [src/morie] > out.tsv
"""

from __future__ import annotations

import ast
import importlib
import inspect
import math
import re
import sys
from pathlib import Path

INF = math.inf
# (values sorted ascending, p, R's quantile(values, p, type = 7))
QCASES = [
    ([1.0, 2.0, INF], 0.5, 2.0),  # weight 0 on the infinite neighbour
    ([1.0, 2.0, INF], 0.75, INF),
    ([-INF, 1.0, 2.0], 0.25, -INF),
    ([-INF, 1.0], 0.5, -INF),
    ([INF, INF], 0.5, INF),
    ([-1e308, 1e308], 0.5, 0.0),  # the difference overflows
    ([1.0, 2.0, 3.0, 4.0], 0.5, 2.5),  # finite control
]
# (values, R's median(values))
MCASES = [([-INF, 1.0], -INF), ([1.0, INF], INF), ([1.0, 2.0, INF], 2.0), ([1.0, 2.0, 3.0, 4.0], 2.5)]
# (values, R's IQR(values))
ICASES = [([1.0, 2.0, 3.0, INF], INF), ([1.0, 2.0, 3.0, 4.0], 1.5), ([-INF, 1.0, 2.0, 3.0], INF)]

NAME = re.compile(r"(quantile|median|percentile|iqr|q7)", re.I)
SKIP = re.compile(
    r"(normal|norm_|gamma|beta|chi2|student|_t_|gev|gpd|evt|mixture|weighted|boot|"
    r"filter|frequency|test|_se$|unbiased|rule|prior|kernel|log_tail|grid|local_linear|"
    r"bayesian|bnp|ci_|_ci|clr|time|trend)",
    re.I,
)


class _NotScalarError(Exception):
    pass


def same(a, b):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return False
    if math.isnan(b):
        return math.isnan(a)
    if math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def scalar(r):
    if isinstance(r, (list, tuple)) and len(r) == 1:
        r = r[0]
    if hasattr(r, "tolist"):
        r = r.tolist()
        if isinstance(r, list) and len(r) == 1:
            r = r[0]
    return r


def kind(name):
    n = name.lower()
    if "iqr" in n:
        return "iqr"
    if "median" in n:
        return "median"
    if "percentile" in n:
        return "percentile"
    return "quantile"


def probe(fn, name):
    k = kind(name)
    try:
        sig = inspect.signature(fn)
        req = [
            p
            for p in sig.parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return "not probed", "no signature"
    bad = []

    def num(got):
        try:
            return float(got)
        except (TypeError, ValueError):
            raise _NotScalarError(type(got).__name__) from None

    try:
        if k in ("median", "iqr"):
            if len(req) != 1:
                return "not probed", f"{len(req)} required arguments"
            for v, want in MCASES if k == "median" else ICASES:
                got = num(scalar(fn(list(v))))
                if not same(got, want):
                    bad.append(f"{v}->{got!r} (R {want!r})")
        else:
            if len(req) != 2:
                return "not probed", f"{len(req)} required arguments"
            for v, p, want in QCASES:
                pp = p * 100 if k == "percentile" else p
                got = num(scalar(fn(list(v), pp)))
                if not same(got, want):
                    bad.append(f"{v},{p}->{got!r} (R {want!r})")
    except _NotScalarError as e:
        return "not probed", f"returns {e}"
    except Exception as e:  # noqa: BLE001 - a helper that refuses Inf is reported, not fixed here
        return "error", f"{type(e).__name__}: {str(e)[:80]}".replace("\n", " ")
    return ("differs", "; ".join(bad)) if bad else ("matches R", "")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "src/morie")
    sys.path.insert(0, str(root.parent))
    rows = []
    for f in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        names = [
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and NAME.search(n.name) and not SKIP.search(n.name)
        ]
        if not names:
            continue
        mod = ".".join(f.relative_to(root.parent).with_suffix("").parts)
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            rows += [(f"{f}", n, "not probed", f"import: {type(e).__name__}") for n in names]
            continue
        for n in names:
            st, why = probe(getattr(m, n), n)
            rows.append((str(f), n, st, why))
    print("file\tfunction\tstatus\tdetail")
    for r in rows:
        print("\t".join(r))
    from collections import Counter

    print("#", dict(Counter(r[2] for r in rows)), file=sys.stderr)


if __name__ == "__main__":
    main()
