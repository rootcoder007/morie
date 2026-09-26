# SPDX-License-Identifier: AGPL-3.0-or-later
"""MRM-framework analyses on Ontario SIU (Special Investigations Unit) data.

Python parity for `mrm_siu_*` (see `r-package/morie/R/mrm_siu.R`).

Functions:
    mrm_siu_case_to_decision_km: KM-style summary of the gap from
        date_of_incident_iso to date_of_director_decision_iso, with
        right-censoring of still-open cases.
    mrm_siu_per_service_rate: per-(service, year) case counts.
    mrm_siu_outcome_classifier: tabulates Director's-decision outcomes
        by service with within-service share.

Unlike OTIS (no placement dates) and TPS (no per-person ID), SIU
exposes per-case dates with a stable `police_service` jurisdiction,
enabling a real time-to-outcome analysis. This is the analysis the
MA-thesis "210-day TTR" claim should have been.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = [
    "mrm_siu_case_to_decision_km",
    "mrm_siu_per_service_rate",
    "mrm_siu_outcome_classifier",
]


@dataclass
class SIUCaseDecisionResult:
    pooled: pd.DataFrame
    by_service: pd.DataFrame


def _km_summary(time, event, probs=(0.25, 0.5, 0.75)):
    """Kaplan-Meier summary as survival::survfit.

    Quantiles by quantile.survfit's rule (first time the cumulative
    incidence reaches p, the midpoint when the curve sits exactly on
    1 - p) and the restricted mean survival to the last time
    (print.survfit's rmean).  Mirrors ``.mrm_km_summary`` in the R arm.
    """
    time = [float(v) for v in time]
    event = [bool(v) for v in event]
    ut = sorted(set(time))
    surv, s = [], 1.0
    for u in ut:
        n_risk = sum(t >= u for t in time)
        n_ev = sum(t == u and e for t, e in zip(time, event))
        s *= 1 - n_ev / n_risk
        surv.append(s)
    x = [min(0.0, ut[0]), *ut]
    y = [0.0, *(1 - v for v in surv)]
    xmax = x[-1]
    seen, xs, ys = set(), [], []
    for a, b in zip(x, y):
        if b not in seen:
            seen.add(b)
            xs.append(a)
            ys.append(b)
    tol = 2.220446049250313e-16**0.5
    q = []
    for p in probs:
        if max(ys) < p:
            q.append(float("nan"))
            continue
        i1 = next(i for i, v in enumerate(ys) if v + tol >= p)
        i2 = next((i for i, v in enumerate(ys) if v - tol >= p), None)
        if abs(p - ys[-1]) < tol:
            q.append((xs[i1] + xmax) / 2)
        else:
            q.append((xs[i1] + xs[i2]) / 2)
    left = [1.0, *surv[:-1]]
    rmean = sum(sl * (b - a) for sl, a, b in zip(left, [0.0, *ut[:-1]], ut))
    return q, rmean


def _summarise(gap, cens, label) -> dict:
    """Kaplan-Meier summary of case-to-decision days; open cases censored.

    ``median_days`` / ``p25_days`` / ``p75_days`` are KM quantiles of the
    time to decision and ``mean_days`` the restricted mean to the last
    follow-up, all as survival::survfit (see :func:`_km_summary`).
    """
    if gap.size == 0:
        return {
            "stratum": label,
            "n": 0,
            "n_censored": 0,
            "median_days": np.nan,
            "mean_days": np.nan,
            "p25_days": np.nan,
            "p75_days": np.nan,
            "max_days": np.nan,
        }
    (q25, q50, q75), rmean = _km_summary(gap, [not c for c in cens])
    return {
        "stratum": label,
        "n": int(gap.size),
        "n_censored": int(cens.sum()),
        "median_days": q50,
        "mean_days": round(rmean, 2),
        "p25_days": q25,
        "p75_days": q75,
        "max_days": float(gap.max()),
    }


def mrm_siu_case_to_decision_km(
    data: pd.DataFrame,
    *,
    incident_col: str = "date_of_incident_iso",
    decision_col: str = "date_of_director_decision_iso",
    service_col: str = "police_service",
    censor_open_cases: bool = True,
    min_n: int = 5,
) -> SIUCaseDecisionResult:
    """KM-style time-from-incident-to-Director's-decision summary."""
    df = pd.coerce_frame(data)

    def _day(v):
        # ISO date -> day ordinal; missing or unparseable -> None
        try:
            return datetime.date.fromisoformat(str(v)[:10]).toordinal()
        except (TypeError, ValueError):
            return None

    inc = [_day(v) for v in df[incident_col]]
    dec = [_day(v) for v in df[decision_col]]
    svc = [str(v) for v in df[service_col]]
    gap = [d - i if i is not None and d is not None else None for i, d in zip(inc, dec)]
    censored = [False] * len(gap)
    if censor_open_cases:
        # open cases: right-censored at the latest decision date in the data
        cutoff = max(d for d in dec if d is not None)
        for k, (i, d) in enumerate(zip(inc, dec)):
            if i is not None and d is None:
                gap[k] = cutoff - i
                censored[k] = True
    ok = [g is not None and g >= 0 for g in gap]
    gap_v = np.array([float(g) for g, o in zip(gap, ok) if o])
    svc_v = [v for v, o in zip(svc, ok) if o]
    cens_v = np.array([c for c, o in zip(censored, ok) if o])
    pooled = pd.DataFrame([_summarise(gap_v, cens_v, "pooled")])

    rows = []
    for sv in dict.fromkeys(svc_v):
        if not sv or sv in ("nan", "None"):
            continue
        idx = [k for k, v in enumerate(svc_v) if v == sv]
        if len(idx) < min_n:
            continue
        rows.append(_summarise(np.array([gap_v[k] for k in idx]), np.array([cens_v[k] for k in idx]), sv))
    by_service = pd.DataFrame(rows)

    return SIUCaseDecisionResult(pooled=pooled, by_service=by_service)


def mrm_siu_per_service_rate(
    data: pd.DataFrame,
    *,
    service_col: str = "police_service",
    incident_col: str = "date_of_incident_iso",
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """Per-police-service case counts by year (and optional stratum)."""
    df = pd.coerce_frame(data).copy()
    df["_year"] = pd.to_datetime(df[incident_col], errors="coerce").dt.year
    svc = df[service_col].astype(str)
    df = df[df["_year"].notna() & (svc.str.len() > 0) & (svc != "nan")]
    cols = [service_col, "_year"]
    if stratify_col is not None:
        df = df[df[stratify_col].notna()]
        cols.append(stratify_col)
    out = df.groupby(cols, dropna=False).size().reset_index(name="n_cases")
    out = out[out["n_cases"] > 0]
    out = out.rename(columns={service_col: "service", "_year": "year"})
    if stratify_col is not None:
        out = out.rename(columns={stratify_col: "stratum"})
    return out.reset_index(drop=True)


def mrm_siu_outcome_classifier(
    data: pd.DataFrame,
    *,
    outcome_col: str = "director_decision_category",
    service_col: str = "police_service",
) -> pd.DataFrame:
    """Tabulate SIU Director's-decision outcomes by service."""
    df = pd.coerce_frame(data).copy()
    if outcome_col not in df.columns:
        for alt in [
            "director_decision",
            "outcome",
            "decision",
            "director_decision_outcome",
            "director_decision_text",
            "charges_recommended",
            "directors_decision_reasonable",
        ]:
            if alt in df.columns:
                outcome_col = alt
                break
    if outcome_col not in df.columns:
        raise KeyError("No outcome column found in SIU data.")
    out = df[outcome_col].astype(str)
    svc = df[service_col].astype(str)
    ok = (out.str.len() > 0) & (out != "nan") & (svc.str.len() > 0) & (svc != "nan")
    tbl = (
        pd.crosstab(svc[ok], out[ok]).reset_index().melt(id_vars=service_col, var_name="outcome", value_name="n_cases")
    )
    totals = tbl.groupby(service_col)["n_cases"].transform("sum")
    tbl["share_within_service"] = (tbl["n_cases"] / totals).round(4)
    tbl = tbl[tbl["n_cases"] > 0].rename(columns={service_col: "service"})
    return tbl.reset_index(drop=True)
