# SPDX-License-Identifier: AGPL-3.0-or-later
"""Distributional drift between two CSVs: ``cpsdrft()`` and ``benfordtst()``.

Same statistics and decisions as rmoriebricklayer: per shared column a
two-sample Kolmogorov-Smirnov test (asymptotic p-value), the population
stability index over the reference's deciles and the Jensen-Shannon
divergence for numeric and date columns; a chi-square test of homogeneity,
PSI and JS over the reference's categories for categorical ones;
identifier-like columns get no verdict. Pure Python.

One difference: when a categorical table has expected counts below 5 the
chi-square p-value is a Monte Carlo one (10,000 tables with the same
margins), and morie draws those tables from Python's generator, so that
p-value differs from R's by Monte Carlo error.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re

from . import _csv

__all__ = ["drift", "benford", "ks", "psi"]


def _chisq_sf(stat: float, df: float) -> tuple[float, float]:
    from morie.fn import _stats_core as st

    return float(st.chi2.sf(stat, df)), float(st.chi2.logsf(stat, df))


def ks(x: list[float], y: list[float]) -> dict:
    """Two-sample KS: ``statistic`` (largest ECDF gap), asymptotic ``p_value``, ``neff``."""
    a, b = sorted(x), sorted(y)
    nx, ny = len(a), len(b)
    if not nx or not ny:
        raise ValueError("both samples need at least one value")
    i = j = 0
    d = 0.0
    while i < nx and j < ny:
        v = min(a[i], b[j])
        while i < nx and a[i] == v:
            i += 1
        while j < ny and b[j] == v:
            j += 1
        d = max(d, abs(i / nx - j / ny))
    ne = nx * ny / (nx + ny)
    return {"statistic": d, "p_value": _ks_p(d, ne), "neff": ne}


def _ks_p(d: float, ne: float) -> float:
    if not d > 0 or not ne > 0:
        return 1.0
    t = math.sqrt(ne) * d
    if t < 1e-12:
        return 1.0
    s = 0.0
    if t < 1.0:
        z = -(math.pi * math.pi / 8.0) / (t * t)
        w = math.log(t)
        for k in range(1, 200, 2):
            term = math.exp(k * k * z - w)
            s += term
            if term < 1e-300:
                break
        p = 1.0 - math.sqrt(2.0 * math.pi) * s
    else:
        for k in range(1, 201):
            term = math.exp(-2.0 * k * k * t * t)
            s += term if k % 2 else -term
            if term < 1e-300:
                break
        p = 2.0 * s
    return min(1.0, max(0.0, p))


def _psi_js(p: list[float], q: list[float], eps: float) -> tuple[float, float]:
    s = 0.0
    for a, b in zip(p, q):
        a, b = max(a, eps), max(b, eps)
        s += (a - b) * math.log(a / b)
    js = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0:
            js += 0.5 * a * math.log(a / m)
        if b > 0:
            js += 0.5 * b * math.log(b / m)
    return s, max(js, 0.0)


def _q7(v: list[float], p: float) -> float:
    n = len(v)
    if n == 1:
        return v[0]
    h = (n - 1) * p
    lo = math.floor(h)
    if lo >= n - 1:
        return v[-1]
    return v[lo] + (h - lo) * (v[lo + 1] - v[lo])


def psi(x: list[float], y: list[float], bins: int = 10, eps: float = 1e-6) -> dict:
    """PSI and Jensen-Shannon divergence of ``y`` against the reference ``x``'s quantile bins."""
    x = [float(f"{v:.12g}") for v in x]
    y = [float(f"{v:.12g}") for v in y]
    sx = sorted(x)
    probs = [0.0] + [i * (1.0 / bins) for i in range(1, bins)] + [1.0]
    q = []
    for p in probs:
        v = _q7(sx, p)
        if v not in q:
            q.append(v)
    edges = [-math.inf, *q[1:-1], math.inf] if len(q) >= 3 else [-math.inf, *q, math.inf]

    def props(vals):
        import bisect

        c = [0] * (len(edges) - 1)
        for v in vals:
            c[bisect.bisect_left(edges, v) - 1] += 1  # right-closed bins (lo, hi], as cut()
        return [k / len(vals) for k in c]

    s, js = _psi_js(props(x), props(y), eps)
    return {"psi": s, "jsdiv": js}


def _counts(v: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in v:
        out[s] = out.get(s, 0) + 1
    return out


def _homog(av: list[str], bv: list[str], seed: int) -> dict:
    cx, cy = _counts(av), _counts(bv)
    lv = sorted(set(cx) | set(cy))
    x = [cx.get(k, 0) for k in lv]
    y = [cy.get(k, 0) for k in lv]
    nx, ny = sum(x), sum(y)
    pooled = [(a + b) / (nx + ny) for a, b in zip(x, y)]
    keep = [i for i, p in enumerate(pooled) if p > 0]
    ex = [nx * pooled[i] for i in keep]
    ey = [ny * pooled[i] for i in keep]
    stat = sum((x[i] - e) ** 2 / e for i, e in zip(keep, ex)) + sum((y[i] - e) ** 2 / e for i, e in zip(keep, ey))
    df = len(keep) - 1
    if df < 1:
        return {"statistic": stat, "df": 0, "p_value": None, "logpvalue": None, "method": "inapplicable: one category"}
    if (nx + ny) / len(keep) < 2:
        return {
            "statistic": stat,
            "df": df,
            "p_value": None,
            "logpvalue": None,
            "method": f"inapplicable: {len(keep)} categories for {nx + ny} observations (fewer than two per category)",
        }
    if min(ex + ey) < 5:
        from morie.fn.chisqmc import chisq_monte_carlo

        p = float(chisq_monte_carlo([[x[i] for i in keep], [y[i] for i in keep]], B=10000, seed=seed)["p_value"])
        return {
            "statistic": stat,
            "df": df,
            "p_value": p,
            "logpvalue": math.log(p),
            "method": "Monte Carlo permutation (small expected counts)",
        }
    p, lp = _chisq_sf(stat, df)
    return {"statistic": stat, "df": df, "p_value": p, "logpvalue": lp, "method": "chi-square approximation"}


_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_DTIME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}(:[0-9]{2})?")
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _as_dates(v: list) -> list | None:
    """Days (dates) or seconds (date-times, UTC) since 1970 for a column of ISO strings; else None."""
    ok = [s for s in v if s is not None]
    if not ok or not all(isinstance(s, str) for s in ok):
        return None
    try:
        if all(_DATE.match(s) for s in ok):
            return [None if s is None else (datetime.date.fromisoformat(s) - datetime.date(1970, 1, 1)).days for s in v]
        if all(_DTIME.match(s) for s in ok):
            out = []
            for s in v:
                if s is None:
                    out.append(None)
                    continue
                m = re.match(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(:\d{2})?)", s)
                t = m.group(2) if m.group(3) else m.group(2) + ":00"
                dt = datetime.datetime.fromisoformat(f"{m.group(1)} {t}").replace(tzinfo=datetime.timezone.utc)
                out.append((dt - _EPOCH).total_seconds())
            return out
    except ValueError:
        return None
    return None


def _row(nm, typ, **kw) -> dict:
    r = {
        "column": nm,
        "type": typ,
        "statistic": None,
        "p_value": None,
        "psi": None,
        "jsdiv": None,
        "unseen_share": None,
        "drifted": None,
        "note": "",
    }
    r.update(kw)
    return r


def _rtext(v) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return f"{v:.15g}"
    return str(v)


def _pinned_csv(d: str, provenance_file: str) -> str:
    from ._prov import safe_rel

    ppath = safe_rel(provenance_file, d)
    if ppath is None:
        raise ValueError(f"'{provenance_file}' is not a plain relative path inside the capsule")
    with open(ppath, encoding="utf-8") as fh:
        res = json.load(fh).get("resource") or {}
    p = safe_rel(res.get("filename"), d)
    if p is None or not os.path.isfile(p):
        raise FileNotFoundError(f"the provenance names no data file inside {d}")
    return p


def drift(
    ref,
    current,
    provenance_file: str = "data_provenance.json",
    alpha: float = 0.01,
    psi_threshold: float = 0.25,
    psi_min_n: int = 1000,
    bins: int = 10,
    id_levels: int = 100,
    seed: int = 1,
) -> dict:
    """Compare a reference CSV with a current one column by column, as ``cpsdrft()``.

    ``ref`` is a capsule directory (its pinned CSV, ``resource.filename`` of
    ``provenance_file``), a CSV path, or a ``{column: (type, values)}`` frame;
    ``current`` a CSV path or a frame. Returns ``columns`` (one row per
    shared column: ``column``, ``type``, ``statistic``, ``p_value``, ``psi``,
    ``jsdiv``, ``unseen_share``, ``drifted``, ``note``), ``added``,
    ``removed``, ``nref``, ``ncur``, ``alpha`` and ``anydrft``, the fields of
    ``cpsdrft()``. A column has drifted when its p-value is below ``alpha``
    or, with at least ``psi_min_n`` values on both sides, PSI exceeds
    ``psi_threshold``; a categorical column with more than ``id_levels``
    distinct values is an identifier and gets no verdict.
    """
    if not 0 < alpha < 1:
        raise ValueError("`alpha` must be a single value strictly inside (0, 1)")
    if psi_min_n < 0:
        raise ValueError("`psi_min_n` must be a single non-negative integer")
    if id_levels < 2:
        raise ValueError("`id_levels` must be a single integer of at least 2")
    if bins < 2:
        raise ValueError("`bins` must be one whole number of at least 2")
    psithr, psiminn, idelvls = psi_threshold, psi_min_n, id_levels
    if not isinstance(ref, dict) and os.path.isdir(ref):
        ref = _pinned_csv(os.fspath(ref), provenance_file)
    rf = _csv.frame(ref) if not isinstance(ref, dict) else ref
    cf = _csv.frame(current) if not isinstance(current, dict) else current
    shared = [c for c in rf if c in cf]
    added = [c for c in cf if c not in rf]
    removed = [c for c in rf if c not in cf]
    rows = []
    for nm in shared:
        (ta, a), (tb, b) = rf[nm], cf[nm]
        numeric = ta in ("integer", "numeric") and tb in ("integer", "numeric")
        typ = "numeric"
        if not numeric and ta == "character" and tb == "character":
            da, db = _as_dates(a), _as_dates(b)
            if da is not None and db is not None:
                a, b, numeric, typ = da, db, True, "date"
        if numeric:
            av = [float(v) for v in a if v is not None and v == v]
            bv = [float(v) for v in b if v is not None and v == v]
            if not av or not bv:
                rows.append(_row(nm, typ, note="no non-missing values on one side"))
                continue
            k = ks(av, bv)
            p = psi(av, bv, bins=bins)
            flag = len(av) >= psiminn and len(bv) >= psiminn and p["psi"] > psithr
            rows.append(
                _row(
                    nm,
                    typ,
                    statistic=k["statistic"],
                    p_value=k["p_value"],
                    psi=p["psi"],
                    jsdiv=p["jsdiv"],
                    drifted=k["p_value"] < alpha or flag,
                )
            )
            continue
        av = [_rtext(v) for v in a if v is not None]
        bv = [_rtext(v) for v in b if v is not None]
        if not av or not bv:
            rows.append(_row(nm, "categorical", note="no non-missing values on one side"))
            continue
        seen = set(av)
        unseen = sum(1 for v in bv if v not in seen) / len(bv)
        n_lv = len(seen | set(bv))
        if n_lv > idelvls or (n_lv >= 20 and n_lv > 0.2 * (len(av) + len(bv))):
            rows.append(
                _row(
                    nm,
                    "identifier",
                    unseen_share=unseen,
                    note=f"{n_lv} distinct values: distribution tests do not apply",
                )
            )
            continue
        cs = _homog(av, bv, seed)
        lv = list(dict.fromkeys(av))
        ca, cb = _counts(av), _counts(bv)
        px = [ca.get(k, 0) / len(av) for k in lv] + [0.0]
        py = [cb.get(k, 0) / len(bv) for k in lv] + [unseen]
        ps, js = _psi_js(px, py, 1e-6)
        flag = len(av) >= psiminn and len(bv) >= psiminn and ps > psithr
        pc = cs["p_value"]
        rows.append(
            _row(
                nm,
                "categorical",
                statistic=cs["statistic"],
                p_value=pc,
                psi=ps,
                jsdiv=js,
                unseen_share=unseen,
                drifted=(True if flag else None) if pc is None else (pc < alpha or flag),
                note=cs["method"] if pc is None else "",
            )
        )
    nref = len(next(iter(rf.values()))[1]) if rf else 0
    ncur = len(next(iter(cf.values()))[1]) if cf else 0
    return {
        "columns": rows,
        "added": added,
        "removed": removed,
        "nref": nref,
        "ncur": ncur,
        "alpha": alpha,
        "anydrft": any(r["drifted"] is True for r in rows) or bool(added) or bool(removed),
    }


def _digit(v: float) -> int:
    v = abs(v)
    if v < 2.2250738585072014e-308:
        for prec in range(17):
            buf = f"{v:.{prec}e}"
            if float(buf) == v:
                break
    else:
        buf = f"{v:.14e}"
    return min(9, max(1, ord(buf[0]) - 48))


def benford(x) -> dict:
    """First-digit test against Benford's law, as ``benfordtst()``: ``counts``,
    ``expected``, ``proportion`` (digits ``"1"``..``"9"``), ``statistic``, ``df``
    (8), ``p_value``, ``logpvalue`` and ``n``, over the finite non-zero values."""
    vals = [float(v) for v in x if v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v != 0]
    if not vals:
        raise ValueError("`x` has no finite non-zero values to take a leading digit from")
    counts = [0] * 9
    for v in vals:
        counts[_digit(v) - 1] += 1
    n = sum(counts)
    exp = [n * math.log10(1 + 1 / d) for d in range(1, 10)]
    stat = sum((c - e) ** 2 / e for c, e in zip(counts, exp))
    p, lp = _chisq_sf(stat, 8)
    keys = [str(d) for d in range(1, 10)]
    return {
        "counts": dict(zip(keys, counts)),
        "expected": dict(zip(keys, exp)),
        "proportion": dict(zip(keys, (c / n for c in counts))),
        "statistic": stat,
        "df": 8,
        "p_value": p,
        "logpvalue": lp,
        "n": n,
    }
