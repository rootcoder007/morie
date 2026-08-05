# morie.fn -- function file (rootcoder007/morie)
"""Event-study leads + lags coefficients."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["event_study_coefficients"]


def event_study_coefficients(y, D, unit, time, cohort, max_lead=None,
                             max_lag=None, ref=-1):
    """
    Event-study leads + lags coefficients

    Formula: the dynamic two-way fixed effects specification

        y_it = alpha_i + lambda_t + sum_{e != ref} mu_e 1{t - g_i = e}
               + eps_it

    where g_i is unit i's treatment cohort and e = t - g_i is event time.
    One event-time indicator must be dropped for identification against
    the unit and period effects; e = ref (conventionally -1, the period
    before treatment) is the omitted category, so every mu_e is read
    relative to it.  Never-treated units (cohort NaN/NA/infinite) enter
    only through alpha_i and lambda_t and so act as a clean control
    group.

    CAVEAT, not a defect of the code: with heterogeneous, dynamic
    treatment effects the mu_e from this regression are contaminated --
    they are weighted sums of cohort-specific effects at OTHER relative
    periods, with weights that can be negative.  Sun & Abraham (2021)
    prove this and give the interaction-weighted alternative.  The
    specification here is the one the docstring states; read it with
    that caveat in mind.

    Parameters
    ----------
    y : array-like
        Outcome, one entry per unit-period.
    D : array-like
        Treatment indicator (carried through for reference; the event
        dummies are built from ``cohort`` and ``time``).
    unit, time : array-like
        Unit and period labels.
    cohort : array-like
        First treated period per observation; NaN/NA/inf marks a
        never-treated unit.
    max_lead, max_lag : int, optional
        Truncate the event window to [-max_lead, max_lag].
    ref : int
        Omitted event time.

    Returns
    -------
    result : dict
        Keys: estimate (coefficient at e = 0), event_times, coef, se,
        sigma2, resid_df, n_units, n_periods, n, method.

    References
    ----------
    Sun & Abraham (2021), Journal of Econometrics 225(2):175-199,
    doi:10.1016/j.jeconom.2020.09.006.
    Borusyak & Jaravel (2017), "Revisiting Event Study Designs",
    working paper.
    """
    y = [float(v) for v in y]
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    D = [float(v) for v in D]
    unit = list(unit)
    time = [float(v) for v in time]
    coh = [float(v) for v in cohort]
    if not (len(D) == len(unit) == len(time) == len(coh) == n):
        raise ValueError("y, D, unit, time and cohort must have equal length")
    us = sorted(set(str(v) for v in unit))
    ts = sorted(set(time))
    U = len(us)
    Tn = len(ts)
    if U < 2 or Tn < 2:
        raise ValueError("need at least two units and two periods")
    ui = {v: i for i, v in enumerate(us)}
    ti = {v: i for i, v in enumerate(ts)}
    ev = []
    for i in range(n):
        g = coh[i]
        ev.append(None if (g != g or g == float("inf")) else int(round(time[i] - g)))
    seen = sorted(set(e for e in ev if e is not None))
    if max_lead is not None:
        seen = [e for e in seen if e >= -int(max_lead)]
    if max_lag is not None:
        seen = [e for e in seen if e <= int(max_lag)]
    ref = int(ref)
    ets = [e for e in seen if e != ref]
    if not ets:
        raise ValueError("no event-time indicators left after dropping ref")
    p = 1 + (U - 1) + (Tn - 1) + len(ets)
    if n <= p:
        raise ValueError("more parameters (%d) than observations (%d)" % (p, n))
    X = []
    for i in range(n):
        row = [1.0]
        row += [1.0 if ui[str(unit[i])] == k else 0.0 for k in range(1, U)]
        row += [1.0 if ti[time[i]] == k else 0.0 for k in range(1, Tn)]
        row += [1.0 if ev[i] == e else 0.0 for e in ets]
        X.append(row)
    b = core.lstsq(X, y, 0.0)
    fit = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
    resid = [y[i] - fit[i] for i in range(n)]
    dof = n - p
    s2 = sum(r * r for r in resid) / dof
    XtX = [[sum(X[i][a] * X[i][c] for i in range(n)) for c in range(p)]
           for a in range(p)]
    off = 1 + (U - 1) + (Tn - 1)
    coef = []
    se = []
    for k, e in enumerate(ets):
        j = off + k
        col = core.cholsolve(XtX, [1.0 if q == j else 0.0 for q in range(p)])
        coef.append(b[j])
        se.append(math.sqrt(s2 * col[j]))
    est = coef[ets.index(0)] if 0 in ets else float("nan")
    return RichResult(payload={
        "estimate": est,
        "event_times": ets,
        "coef": coef,
        "se": se,
        "sigma2": s2,
        "resid_df": dof,
        "n_units": U,
        "n_periods": Tn,
        "n": n,
        "method": "Event-study leads + lags coefficients",
    })


def cheatsheet():
    return "evstud: Event-study leads + lags coefficients"


# compact alias per ledger/NAMING.md
eventstudycoefficients = event_study_coefficients
