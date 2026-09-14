# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MRM research framework: load, reconcile, report.

Parity with rmorie's R/mrm_flagship.R. Multilevel Reconciliation
Methodology -- load a special-investigations dataset with provenance,
reconcile it against a second source under an explicit matching schema,
and render a publication-ready table with morie's own renderer.

Reconciliation is the part that earns its keep. Two sources describing
the same events rarely agree, and the disagreement is the finding: a
match rate, the orphans on each side, and the field-level conflicts
among the records that did match. Silently inner-joining and reporting
the survivors hides all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from morie.fn import _frame_core as pd
from morie.fn._sha2 import sha256 as _sha256

__all__ = [
    "MrmDataset",
    "MrmEffect",
    "MrmReconciliation",
    "mrm_load_si_dataset",
    "mrm_reconcile",
    "mrm_report",
    "mrm_estimate_causal_effect",
]


# ------------------------------------------------------------------ loading

@dataclass
class MrmDataset:
    data: object
    provenance: dict

    def __repr__(self) -> str:
        p = self.provenance
        return ("MRM dataset %r: %d rows x %d cols, sha256 %s"
                % (p.get("name"), p.get("n_rows"), p.get("n_cols"),
                   str(p.get("sha256"))[:16] + "..."))


def _rows_and_cols(data):
    """Row count and column names for a frame-like or rows of mappings."""
    if hasattr(data, "columns"):
        cols = list(data.columns)
        n = len(data[cols[0]]) if cols else 0
        return n, cols, lambda c: list(data[c])
    rows = list(data)
    cols = list(rows[0].keys()) if rows else []
    return len(rows), cols, lambda c: [r.get(c) for r in rows]


def _csv_bytes(data) -> str:
    """The data as CSV text, for a checksum that is stable across runs."""
    n, cols, get = _rows_and_cols(data)
    colvals = {c: get(c) for c in cols}
    lines = [",".join(str(c) for c in cols)]
    for i in range(n):
        lines.append(",".join("" if colvals[c][i] is None
                              else str(colvals[c][i]) for c in cols))
    return "\n".join(lines)


def mrm_load_si_dataset(name: str = "otis_b01") -> MrmDataset:
    """Load a bundled sample inside a provenance envelope.

    The checksum is of the data as CSV, so an analysis records exactly
    which snapshot it consumed rather than merely which name it asked
    for.
    """
    data = __import__("morie").load_sample(name)
    n, cols, _ = _rows_and_cols(data)
    digest = _sha256(_csv_bytes(data).encode("utf-8"))
    if isinstance(digest, (bytes, bytearray)):
        digest = digest.hex()
    return MrmDataset(
        data=data,
        provenance={
            "name": name,
            "n_rows": n,
            "n_cols": len(cols),
            "sha256": digest,
            "loaded_at": datetime.now(timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        })


# ------------------------------------------------------------ reconciliation

@dataclass
class MrmReconciliation:
    matched: list[dict]
    unmatched_primary: list[dict]
    unmatched_secondary: list[dict]
    conflicts: list[dict] = field(default_factory=list)
    match_rate: float | None = None
    schema: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return ("MRM reconciliation\n"
                "  keys       : %s\n"
                "  match rate : %.1f%% (%d matched, %d/%d orphans)\n"
                "  conflicts  : %d field-level"
                % (", ".join(self.schema.get("keys", [])),
                   100 * (self.match_rate or 0.0), len(self.matched),
                   len(self.unmatched_primary),
                   len(self.unmatched_secondary), len(self.conflicts)))

    def conflicts_frame(self):
        if not self.conflicts:
            return pd.DataFrame({"key": [], "field": [],
                                 "primary_value": [], "secondary_value": []})
        return pd.DataFrame({k: [c[k] for c in self.conflicts]
                             for k in self.conflicts[0]})


def _as_rows(data, what: str) -> list[dict]:
    if hasattr(data, "data") and hasattr(data, "provenance"):
        data = data.data                      # an MrmDataset
    if hasattr(data, "matched"):
        data = data.matched                   # an MrmReconciliation
    if data is None:
        raise ValueError("`%s` must not be None" % what)
    if hasattr(data, "columns"):
        cols = list(data.columns)
        n = len(data[cols[0]]) if cols else 0
        colvals = {c: list(data[c]) for c in cols}
        return [{c: colvals[c][i] for c in cols} for i in range(n)]
    rows = list(data)
    if rows and not all(hasattr(r, "keys") for r in rows):
        raise ValueError("`%s` must be a frame or rows of mappings" % what)
    return [dict(r) for r in rows]


def mrm_reconcile(primary, secondary, keys, compare=None,
                  numeric_tolerance: float = 0.0) -> MrmReconciliation:
    """Join two sources on `keys` and report how far they disagree.

    `numeric_tolerance` is the absolute slack under which a numeric
    disagreement in a `compare` column is not counted as a conflict --
    two sources recording the same date a day apart, for instance.
    """
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    if not keys:
        raise ValueError("`keys` must name at least one column")
    p_rows = _as_rows(primary, "primary")
    s_rows = _as_rows(secondary, "secondary")
    for rows, what in ((p_rows, "primary"), (s_rows, "secondary")):
        if rows:
            missing = [k for k in keys if k not in rows[0]]
            if missing:
                raise ValueError("`%s` is missing column(s): %s"
                                 % (what, ", ".join(missing)))

    def kof(r):
        return "\r".join(str(r.get(k)) for k in keys)

    s_index: dict[str, list[dict]] = {}
    for r in s_rows:
        s_index.setdefault(kof(r), []).append(r)
    p_keys = {kof(r) for r in p_rows}

    # A bare string is one column name, not a sequence of characters:
    # list("lab") is ['l', 'a', 'b'], which silently compares nothing.
    if isinstance(compare, str):
        compare = [compare]
    compare = list(compare) if compare else []
    matched, conflicts = [], []
    for pr in p_rows:
        k = kof(pr)
        for sr in s_index.get(k, []):
            row = dict(pr)
            for c, v in sr.items():
                if c in keys:
                    continue
                if c in pr:
                    # the same column on both sides: keep both, suffixed,
                    # exactly as the R merge does
                    row.pop(c, None)
                    row["%s.primary" % c] = pr[c]
                    row["%s.secondary" % c] = v
                else:
                    row[c] = v
            matched.append(row)
            for f in compare:
                if f not in pr or f not in sr:
                    continue
                a, b = pr[f], sr[f]
                if a is None and b is None:
                    continue
                bad = True
                if a is not None and b is not None:
                    if isinstance(a, (int, float)) and \
                       isinstance(b, (int, float)) and \
                       not isinstance(a, bool) and not isinstance(b, bool):
                        bad = abs(float(a) - float(b)) > numeric_tolerance
                    else:
                        bad = str(a) != str(b)
                if bad:
                    conflicts.append({
                        "key": "/".join(str(pr.get(x)) for x in keys),
                        "field": f,
                        "primary_value": "" if a is None else str(a),
                        "secondary_value": "" if b is None else str(b)})

    return MrmReconciliation(
        matched=matched,
        unmatched_primary=[r for r in p_rows if kof(r) not in s_index],
        unmatched_secondary=[r for r in s_rows if kof(r) not in p_keys],
        conflicts=conflicts,
        match_rate=(sum(1 for r in p_rows if kof(r) in s_index) / len(p_rows)
                    if p_rows else None),
        schema={"keys": keys, "compare": compare,
                "numeric_tolerance": numeric_tolerance})


# ------------------------------------------------------------------ reporting

def _fmt(v, digits: int) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return ("%%.%df" % digits) % v
    return str(v)


def mrm_report(effect=None, reconciliation=None, dataset=None,
               digits: int = 3, stars: bool = True) -> str:
    """Render the MRM result as a plain-text table.

    morie's own renderer: no external table package, so the output
    cannot drift with someone else's defaults. `effect` is any object
    exposing `results` as rows of mappings with `method`, `estimate`,
    `std_error`, `ci_lower`, `ci_upper` and `p_adjusted`.
    """
    lines: list[str] = ["Multilevel Reconciliation Methodology"]

    if dataset is not None:
        p = getattr(dataset, "provenance", None) or {}
        lines.append("")
        lines.append("Source")
        lines.append("  %-14s %s" % ("dataset", p.get("name", "")))
        lines.append("  %-14s %s rows x %s cols"
                     % ("shape", p.get("n_rows"), p.get("n_cols")))
        lines.append("  %-14s %s" % ("sha256",
                                     str(p.get("sha256", ""))[:32]))

    if reconciliation is not None:
        r = reconciliation
        lines.append("")
        lines.append("Reconciliation")
        lines.append("  %-14s %s" % ("keys",
                                     ", ".join(r.schema.get("keys", []))))
        lines.append("  %-14s %.1f%%" % ("match rate",
                                         100 * (r.match_rate or 0.0)))
        lines.append("  %-14s %d matched, %d/%d orphans"
                     % ("counts", len(r.matched),
                        len(r.unmatched_primary),
                        len(r.unmatched_secondary)))
        lines.append("  %-14s %d" % ("conflicts", len(r.conflicts)))

    if effect is not None:
        rows = getattr(effect, "results", None)
        if rows is None:
            raise ValueError("`effect` must expose `results`")
        rows = _as_rows(rows, "effect.results")
        lines.append("")
        lines.append("Causal effect")
        head = "  %-28s %10s %10s %22s %10s" % (
            "method", "estimate", "std.err", "95% CI", "p.adj")
        lines.append(head)
        lines.append("  " + "-" * (len(head) - 2))
        for r in rows:
            ci = "[%s, %s]" % (_fmt(r.get("ci_lower"), digits),
                              _fmt(r.get("ci_upper"), digits))
            star = ""
            if stars:
                pa = r.get("p_adjusted")
                if pa is not None:
                    star = ("***" if pa < 0.001 else "**" if pa < 0.01
                            else "*" if pa < 0.05 else "")
            lines.append("  %-28s %10s %10s %22s %10s%s" % (
                r.get("method", ""), _fmt(r.get("estimate"), digits),
                _fmt(r.get("std_error"), digits), ci,
                _fmt(r.get("p_adjusted"), digits), star))
        cons = getattr(effect, "consensus", None)
        if cons:
            lines.append("  %-28s %10s %10s" % (
                "consensus (inv-variance)",
                _fmt(cons.get("estimate"), digits),
                _fmt(cons.get("std_error"), digits)))
        if stars:
            lines.append("  signif: *** p<0.001, ** p<0.01, * p<0.05")

    return "\n".join(lines)

# ------------------------------------------------------------------ estimation

@dataclass
class MrmEffect:
    results: list[dict]
    consensus: dict
    correction: str
    spec: dict
    failed: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return ("MRM effect: %d of %d estimators, consensus %.4f (se %.4f)"
                % (len(self.results),
                   len(self.results) + len(self.failed),
                   self.consensus.get("estimate", float("nan")),
                   self.consensus.get("std_error", float("nan"))))

    def results_frame(self):
        if not self.results:
            return pd.DataFrame({})
        return pd.DataFrame({k: [r[k] for r in self.results]
                             for k in self.results[0]})


def _norm_sf(z: float) -> float:
    """Two-sided normal tail, by erf, so no extra dependency."""
    import math
    return math.erfc(abs(z) / math.sqrt(2.0))


_Z975 = 1.959963984540054          # qnorm(0.975)


def mrm_estimate_causal_effect(data, treatment: str, outcome: str,
                               covariates, methods=("matching", "ate",
                                                    "aipw", "dml"),
                               correction: str = "holm",
                               seed: int = 42):
    """Estimate one effect several ways, then correct across the answers.

    Composes morie's own estimators -- nearest-neighbour matching, IPW,
    AIPW and partially-linear DML -- on a single specification, corrects
    the p-values for having asked more than once, and pools by inverse
    variance.

    Running four estimators and quoting the friendliest is the failure
    mode this guards against, which is why every answer is returned
    together with the correction applied across them. An estimator that
    cannot run is recorded in `failed` rather than dropped silently: a
    consensus over an unknown subset is not a consensus.
    """
    if isinstance(methods, str):
        methods = (methods,)
    methods = tuple(methods)
    known = ("matching", "ate", "aipw", "dml")
    bad = [m for m in methods if m not in known]
    if bad:
        raise ValueError("unknown method(s): %s; expected a subset of %s"
                         % (", ".join(bad), ", ".join(known)))
    if isinstance(covariates, str):
        covariates = [covariates]
    covariates = list(covariates)

    rows_in = _as_rows(data, "data")
    if not rows_in:
        raise ValueError("`data` must not be empty")
    missing = [c for c in [treatment, outcome] + covariates
               if c not in rows_in[0]]
    if missing:
        raise ValueError("`data` is missing column(s): %s"
                         % ", ".join(missing))

    tvals = {r[treatment] for r in rows_in}
    if not tvals <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(
            "`%s` must be a binary 0/1 treatment; a categorical treatment "
            "must not be silently coerced, because factor level indices "
            "are not data" % treatment)

    frame = pd.DataFrame({c: [r.get(c) for r in rows_in]
                          for c in rows_in[0]})

    out, failed = [], {}

    def attempt(label, fn):
        try:
            est, se = fn()
        except Exception as exc:                  # noqa: BLE001
            failed[label] = "%s: %s" % (type(exc).__name__, exc)
            return
        if est is None or se is None or se <= 0 or se != se or est != est:
            failed[label] = "returned a non-finite estimate or std error"
            return
        p = _norm_sf(est / se)
        out.append({"method": label, "estimate": float(est),
                    "std_error": float(se),
                    "ci_lower": float(est - _Z975 * se),
                    "ci_upper": float(est + _Z975 * se),
                    "p_value": float(p)})

    if "matching" in methods:
        def _matching():
            from morie.matching import (match_nearest_neighbor,
                                        estimate_att_matched)
            m = match_nearest_neighbor(frame, treatment, covariates)
            pairs = getattr(m, "match_pairs", None)
            if pairs is None:
                raise ValueError("matching returned no pairs")
            a = estimate_att_matched(frame, outcome, treatment, pairs)
            if isinstance(a, tuple):
                return a[0], a[1]
            for e, v in (("att", "se"), ("estimate", "std_error"),
                         ("effect", "standard_error")):
                if getattr(a, e, None) is not None:
                    return getattr(a, e), getattr(a, v, None)
            raise ValueError("matching result exposed no estimate")
        attempt("matching (morie native)", _matching)

    if "ate" in methods:
        def _ate():
            import morie
            from morie.causal import (calculate_ipw_weights,
                                      compute_propensity_scores)
            f2 = frame.copy() if hasattr(frame, "copy") else frame
            f2["_ps"] = list(compute_propensity_scores(frame, treatment,
                                                       covariates))
            f2["_ipw"] = list(calculate_ipw_weights(f2, treatment, "_ps"))
            return morie.estimate_ate(f2, outcome, treatment, "_ipw")
        attempt("ipw ate (morie native)", _ate)

    if "aipw" in methods:
        def _aipw():
            from morie.causal import estimate_aipw
            a = estimate_aipw(frame, treatment=treatment, outcome=outcome,
                              covariates=covariates)
            return (a.get("ate", a.get("estimate")),
                    a.get("se", a.get("std_error")))
        attempt("aipw (morie native)", _aipw)

    if "dml" in methods:
        def _dml():
            from morie.fn.causdml2 import causal_dml_partial_lin
            y = [float(r[outcome]) for r in rows_in]
            d = [float(r[treatment]) for r in rows_in]
            X = [[float(r[c]) for c in covariates] for r in rows_in]
            a = causal_dml_partial_lin(y, d, X, seed=seed)
            if isinstance(a, tuple):
                return a[0], a[1]
            pay = getattr(a, "payload", None) or {}
            return (pay.get("theta", getattr(a, "theta", None)),
                    pay.get("se", getattr(a, "se", None)))
        attempt("dml plr (morie native)", _dml)

    if not out:
        first = next(iter(failed.values())) if failed else "no method ran"
        raise RuntimeError("every requested estimator failed; first error: %s"
                           % first)

    p_values = [r["p_value"] for r in out]
    if correction == "none":
        adj = list(p_values)
    else:
        from morie import multiple_testing as mt
        fn = getattr(mt, correction, None)
        if fn is None:
            raise ValueError("unknown correction: %s" % correction)
        res = fn(p_values)
        adj = list(getattr(res, "p_adjusted", getattr(res, "adjusted", res)))
    for r, a in zip(out, adj):
        r["p_adjusted"] = float(a)

    w = [1.0 / r["std_error"] ** 2 for r in out]
    sw = sum(w)
    consensus = {
        "estimate": sum(wi * r["estimate"] for wi, r in zip(w, out)) / sw,
        "std_error": (1.0 / sw) ** 0.5,
    }
    return MrmEffect(results=out, consensus=consensus, correction=correction,
                     spec={"treatment": treatment, "outcome": outcome,
                           "covariates": covariates, "n": len(rows_in)},
                     failed=failed)
