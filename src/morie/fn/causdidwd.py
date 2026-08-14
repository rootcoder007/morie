# morie.fn -- function file (rootcoder007/morie)
r"""Extended two-way fixed effects: TWFE is fine if the model is.

The staggered-adoption literature is often read as "TWFE is broken".
Wooldridge's point is narrower and more useful: TWFE is broken when it
is applied to a model that is **too restrictive**, and there is nothing
wrong with it applied to a sufficiently flexible one. Saturate the
regression in treatment cohort and calendar period and the estimator
recovers the cohort-time treatment effects exactly.

**The algebraic backbone: TWFE is a Mundlak regression.** Including
unit fixed effects in OLS is the same as subtracting unit-specific time
averages -- that is the classical one-way result. The extension proved
here is that adding **both** unit-specific time averages *and*
period-specific cross-sectional averages to a pooled OLS regression
reproduces the **two-way** fixed effects estimates. That is the two-way
Mundlak (TWM) regression, and the equivalence is numerical, not
approximate: same coefficients, to solver precision. The anchor checks
it that way rather than to a tolerance that would hide a different
model.

Why it matters beyond elegance: a full set of two-way fixed effects can
be replaced by a handful of low-dimensional covariates. With many units
this is the difference between a design matrix of size :math:`N + T` and
one of size a few columns.

**The flexible model.** Let :math:`d_g` be a dummy for first treatment
in cohort :math:`g` and :math:`f_t` a period dummy. The saturated
specification interacts every cohort with every post-treatment period,

.. math:: Y_{it} = \alpha_i + \gamma_t
          + \sum_{g}\sum_{t \ge g} \tau_{gt}\, d_{ig} f_{it} + u_{it},

so each :math:`\tau_{gt}` is its own parameter and no cell is forced to
borrow from another. Estimated this way, the coefficients **are** the
:math:`ATT(g,t)`; there is no negative-weight problem because there is
no weighting -- nothing is being averaged. The pathology in the naive
specification comes from imposing a single :math:`\tau` on all cells,
which then has to be recovered as a weighted average, and those weights
can go negative when already-treated units serve as controls.

**Several estimators, one set of numbers.** Section 5 of the paper
establishes that imputation using cohort dummies, pooled OLS on cohort
dummies, random effects, TWFE, and TWFE-based imputation are
*numerically identical* on the flexible model. Two of those routes are
implemented here -- ``etwfe`` (regression) and ``imputation`` -- and the
anchor checks they agree to solver precision, because if they did not
one of them would be wrong.

**Aggregation is a separate, visible choice.** With many cohorts and
periods there are many :math:`\tau_{gt}`, individually imprecise.
Section 7 aggregates them, either to one overall effect or by exposure
time. The weights are the user's to see: ``aggregate`` reports them.

References
----------
Wooldridge, J. M. (2025) "Two-way fixed effects, the two-way Mundlak
regression, and difference-in-differences estimators", *Empirical
Economics* 69, 2545-2587, doi:10.1007/s00181-025-02807-z. Sec. 3 (the
TWFE-TWM equivalence), Sec. 4 (identification of the ATTs under the
usual DiD assumptions and the two-step imputation estimator), Sec. 5
(numerical equivalence of imputation, pooled OLS on cohort dummies,
random effects and TWFE, with Sec. 5.2 on why a flexible TWFE is not
the problem), Sec. 6 (leads and lags) and Sec. 7 (aggregation).

Mundlak, Y. (1978) "On the pooling of time series and cross section
data", *Econometrica* 46(1), 69-85, doi:10.2307/1913646. The device
being extended.

Callaway, B. & Sant'Anna, P. H. C. (2021) "Difference-in-Differences
with multiple time periods", *Journal of Econometrics* 225(2),
200-230, doi:10.1016/j.jeconom.2020.12.001. The 2x2 regression-based
estimator Sec. 6 shows this coincides with.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["two_way_mundlak", "two_way_fixed_effects", "etwfe",
           "imputation", "aggregate"]

_EPS = 1e-12


def _panel(Y, unit, period):
    y = [float(v) for v in k.vec(Y)]
    u = [str(v) for v in unit]
    t = [str(v) for v in period]
    n = len(y)
    if not (len(u) == len(t) == n):
        raise ValueError("causdidwd: Y, unit and period must agree in "
                         "length (%d, %d, %d)" % (n, len(u), len(t)))
    if n < 4:
        raise ValueError("causdidwd: need at least 4 observations, "
                         "got %d" % n)
    return y, u, t, n


def two_way_fixed_effects(Y, unit, period, X):
    r"""Pooled OLS with a full set of unit and period dummies.

    The reference cell is dropped from each set so the design has full
    rank. Returns the coefficients on ``X`` only -- the fixed effects
    are nuisance.
    """
    y, u, t, n = _panel(Y, unit, period)
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("causdidwd: X has %d rows for %d "
                         "observations" % (len(Xm), n))
    us = sorted(set(u))
    ts = sorted(set(t))
    if len(us) < 2 or len(ts) < 2:
        raise ValueError("causdidwd: need at least 2 units and 2 "
                         "periods, got %d and %d" % (len(us), len(ts)))
    ui = {v: i for i, v in enumerate(us)}
    ti = {v: i for i, v in enumerate(ts)}
    p = len(Xm[0])
    rows = []
    for i in range(n):
        r = [float(v) for v in Xm[i]]
        d = [0.0] * (len(us) - 1)
        if ui[u[i]] > 0:
            d[ui[u[i]] - 1] = 1.0
        f = [0.0] * (len(ts) - 1)
        if ti[t[i]] > 0:
            f[ti[t[i]] - 1] = 1.0
        rows.append(r + d + f)
    beta = k.lstsq(k.design(rows, n), y, 1e-10)
    return {"coef": beta[1:1 + p], "full": beta,
            "n_units": len(us), "n_periods": len(ts),
            "n_columns": len(rows[0]) + 1,
            "method": "two-way fixed effects by dummy variables"}


def two_way_mundlak(Y, unit, period, X):
    r"""Pooled OLS with unit-specific and period-specific averages.

    Sec. 3: adding the unit-specific time averages :math:`\bar X_i` and
    the period-specific cross-sectional averages :math:`\bar X_t` to a
    pooled OLS regression reproduces the two-way fixed effects
    estimates **exactly**. The design is far smaller: three blocks of
    width :math:`p` rather than :math:`N + T` dummies.
    """
    y, u, t, n = _panel(Y, unit, period)
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    if len(Xm) != n:
        raise ValueError("causdidwd: X has %d rows for %d "
                         "observations" % (len(Xm), n))
    p = len(Xm[0])
    by_u, by_t = {}, {}
    for i in range(n):
        by_u.setdefault(u[i], []).append(i)
        by_t.setdefault(t[i], []).append(i)
    ubar = {g: [sum(Xm[i][j] for i in idx) / len(idx)
                for j in range(p)] for g, idx in by_u.items()}
    tbar = {g: [sum(Xm[i][j] for i in idx) / len(idx)
                for j in range(p)] for g, idx in by_t.items()}
    rows = [Xm[i] + ubar[u[i]] + tbar[t[i]] for i in range(n)]
    beta = k.lstsq(k.design(rows, n), y, 1e-10)
    return {"coef": beta[1:1 + p], "full": beta,
            "n_columns": 1 + 3 * p,
            "method": "two-way Mundlak: pooled OLS with unit-specific "
                      "time averages and period-specific "
                      "cross-sectional averages; Wooldridge (2025) "
                      "Sec. 3",
            "identical_to": "two-way fixed effects"}


def _cohorts(first_treated, period):
    ts = sorted(set(str(v) for v in period))
    order = {v: i for i, v in enumerate(ts)}
    G = []
    for v in first_treated:
        if v is None:
            G.append(None)
            continue
        s = str(v)
        if s not in order:
            raise ValueError("causdidwd: adoption period %r is not a "
                             "period in the data" % (v,))
        G.append(s)
    if not any(g is not None for g in G):
        raise ValueError("causdidwd: no unit is ever treated")
    return G, ts, order


def etwfe(Y, unit, period, first_treated, X=None):
    r"""The saturated cohort-by-period regression of Sec. 5.

    Every :math:`(g, t)` cell with :math:`t \ge g` gets its own
    coefficient, so the fitted coefficients **are** the
    :math:`ATT(g,t)`. Nothing is averaged, so nothing can be averaged
    with a negative weight.
    """
    y, u, t, n = _panel(Y, unit, period)
    if len(first_treated) != n:
        raise ValueError("causdidwd: %d adoption periods for %d "
                         "observations" % (len(first_treated), n))
    G, ts, order = _cohorts(first_treated, t)
    cells = sorted({(G[i], t[i]) for i in range(n)
                    if G[i] is not None
                    and order[t[i]] >= order[G[i]]})
    if not cells:
        raise ValueError("causdidwd: no post-treatment cell exists")
    us = sorted(set(u))
    ui = {v: i for i, v in enumerate(us)}
    ti = {v: i for i, v in enumerate(ts)}
    Xm = ([[] for _ in range(n)] if X is None
          else [[float(v) for v in r] for r in k.mat(X)])
    rows = []
    for i in range(n):
        cell = [1.0 if (G[i] is not None and (G[i], t[i]) == c)
                else 0.0 for c in cells]
        d = [0.0] * (len(us) - 1)
        if ui[u[i]] > 0:
            d[ui[u[i]] - 1] = 1.0
        f = [0.0] * (len(ts) - 1)
        if ti[t[i]] > 0:
            f[ti[t[i]] - 1] = 1.0
        rows.append(cell + list(Xm[i]) + d + f)
    beta = k.lstsq(k.design(rows, n), y, 1e-10)
    att = {cells[j]: beta[1 + j] for j in range(len(cells))}
    return RichResult(payload={
        "estimate": sum(att.values()) / len(att),
        "att": att, "cells": cells,
        "n_cells": len(cells), "coef": beta,
        "cohorts": sorted({g for g in G if g is not None}),
        "periods": ts, "n": n,
        "method": "extended two-way fixed effects: saturated in "
                  "cohort x period; Wooldridge (2025) Sec. 5",
        "note": "each coefficient IS an ATT(g,t); nothing is averaged, "
                "so no negative weights arise",
    })


def imputation(Y, unit, period, first_treated, X=None):
    r"""The two-step imputation estimator of Sec. 4.

    Fit the two-way model on **untreated** observations only, impute
    the untreated potential outcome everywhere, and average the
    residual within each :math:`(g, t)` cell. Sec. 5 proves this is
    numerically identical to :func:`etwfe`; the anchor checks it.
    """
    y, u, t, n = _panel(Y, unit, period)
    if len(first_treated) != n:
        raise ValueError("causdidwd: %d adoption periods for %d "
                         "observations" % (len(first_treated), n))
    G, ts, order = _cohorts(first_treated, t)
    treated = [G[i] is not None and order[t[i]] >= order[G[i]]
               for i in range(n)]
    untreated = [i for i in range(n) if not treated[i]]
    if len(untreated) < 2:
        raise ValueError("causdidwd: too few untreated observations "
                         "to fit the baseline model")
    us = sorted(set(u))
    ui = {v: i for i, v in enumerate(us)}
    ti = {v: i for i, v in enumerate(ts)}
    Xm = ([[] for _ in range(n)] if X is None
          else [[float(v) for v in r] for r in k.mat(X)])

    def row(i):
        d = [0.0] * (len(us) - 1)
        if ui[u[i]] > 0:
            d[ui[u[i]] - 1] = 1.0
        f = [0.0] * (len(ts) - 1)
        if ti[t[i]] > 0:
            f[ti[t[i]] - 1] = 1.0
        return list(Xm[i]) + d + f

    R = [row(i) for i in untreated]
    beta = k.lstsq(k.design(R, len(R)), [y[i] for i in untreated],
                   1e-10)
    cells = {}
    for i in range(n):
        if not treated[i]:
            continue
        r = [1.0] + row(i)
        yhat = sum(r[j] * beta[j] for j in range(len(beta)))
        cells.setdefault((G[i], t[i]), []).append(y[i] - yhat)
    att = {c: sum(v) / len(v) for c, v in cells.items()}
    return RichResult(payload={
        "estimate": sum(att.values()) / len(att),
        "att": att, "n_cells": len(att),
        "n_untreated_used": len(untreated), "coef": beta,
        "method": "two-step imputation on untreated observations; "
                  "Wooldridge (2025) Sec. 4",
        "identical_to": "etwfe, by Sec. 5",
    })


def aggregate(result, scheme="simple", weights=None):
    r"""Collapse the :math:`ATT(g,t)` to one number or an event-time
    profile, with the weights reported (Sec. 7).
    """
    if scheme not in ("simple", "event", "cohort"):
        raise ValueError("causdidwd: scheme must be simple, event or "
                         "cohort, got %r" % (scheme,))
    att = result["att"] if not isinstance(result, dict) \
        or "att" in result else result
    if not att:
        raise ValueError("causdidwd: nothing to aggregate")
    if scheme == "simple":
        w = ({c: 1.0 / len(att) for c in att} if weights is None
             else dict(weights))
        tot = sum(w.values())
        if abs(tot) <= _EPS:
            raise ValueError("causdidwd: the weights sum to zero")
        return {"estimate": sum(att[c] * w[c] for c in att) / tot,
                "weights": w, "scheme": "simple"}
    keyed = {}
    for (g, t), v in att.items():
        keyed.setdefault(t if scheme == "cohort" else (g, t),
                         []).append(v)
    prof = {kk: sum(vs) / len(vs) for kk, vs in keyed.items()}
    return {"profile": prof, "scheme": scheme,
            "estimate": sum(prof.values()) / len(prof)}


def cheatsheet():
    return ("causdidwd: ETWFE. TWFE == two-way MUNDLAK -- pooled OLS "
            "with unit-specific time averages AND period-specific "
            "cross-sectional averages gives the identical "
            "coefficients, in 3p columns rather than N+T dummies. "
            "TWFE is not broken; a RESTRICTIVE model is. Saturate in "
            "cohort x period and each coefficient IS an ATT(g,t) -- "
            "nothing is averaged, so no negative weights. Imputation "
            "on untreated observations gives the same numbers.")


# compact alias per ledger/NAMING.md
etwfedid = etwfe
