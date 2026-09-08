r"""Two-way fixed effects with heterogeneous treatment effects.

de Chaisemartin, C., & D'Haultfoeuille, X. (2020) "Two-Way Fixed Effects
Estimators with Heterogeneous Treatment Effects", *American Economic
Review* 110(9), 2964-2996.

The paper's result is a warning about a regression everyone runs. The
two-way fixed effects coefficient is *not* an average treatment effect;
it is a weighted sum of the cell-level effects,

.. math::

   \beta_{fe} = E\Bigl[\sum_{(g,t):D_{g,t}=1}
                       \frac{N_{g,t}}{N_1} w_{g,t}\, \Delta_{g,t}\Bigr]
   \tag{Theorem 1}

and the weights :math:`w_{g,t}` -- which come from the residual of
:math:`D` on the two sets of fixed effects -- **can be negative**. A cell
with a negative weight enters the estimate with its effect sign flipped,
so :math:`\beta_{fe}` can be negative when every single
:math:`\Delta_{g,t}` is positive. The weights sum to one, so under
homogeneous effects the problem disappears; heterogeneity is what makes
it bite.

Implemented here:

* :func:`twfe_weights` -- the weights of Theorem 1, from the residual of
  the treatment indicator on group and period fixed effects;
* :func:`twfe` -- the coefficient itself, plus its decomposition and the
  count and mass of the negative weights;
* :func:`did_m` -- the paper's :math:`DID_M` alternative, which compares
  cells that switch treatment between :math:`t-1` and :math:`t` with
  cells whose treatment does not change, and is unbiased for the average
  effect among switchers under common trends whatever the heterogeneity.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causdiddc", "twfe", "twfe_weights", "did_m"]


def _panel(Y, D, group, period):
    Y = [float(v) for v in np.atleast_1d(np.asarray(Y, dtype=float))]
    D = [float(v) for v in np.atleast_1d(np.asarray(D, dtype=float))]
    g = list(group)
    t = list(period)
    n = len(Y)
    if not (len(D) == len(g) == len(t) == n):
        raise ValueError("causdiddc: Y, D, group and period must have "
                         "equal length")
    if n < 4:
        raise ValueError("causdiddc: need at least four observations")
    for v in D:
        if v not in (0.0, 1.0):
            raise ValueError("causdiddc: D must be binary 0/1")
    for v in Y:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("causdiddc: Y contains a non-finite value")
    return Y, D, g, t, n


def _cells(Y, D, g, t):
    """Collapse to (group, period) cells: mean Y, treatment, size."""
    acc = {}
    for i in range(len(Y)):
        key = (g[i], t[i])
        if key not in acc:
            acc[key] = [0.0, 0, D[i]]
        if acc[key][2] != D[i]:
            raise ValueError("causdiddc: treatment varies within the "
                             "(group, period) cell %r" % (key,))
        acc[key][0] += Y[i]
        acc[key][1] += 1
    return dict((k, (v[0] / v[1], v[1], v[2])) for k, v in acc.items())


def twfe_weights(D, group, period, weights=None):
    r"""The :math:`w_{g,t}` of Theorem 1.

    They come from the residual :math:`\varepsilon_{g,t}` of regressing
    the treatment indicator on group and period fixed effects, scaled so
    that they sum to one over the treated cells.
    """
    g, t = list(group), list(period)
    D = [float(v) for v in np.atleast_1d(np.asarray(D, dtype=float))]
    n = len(D)
    if weights is None:
        weights = [1.0] * n
    gs = sorted(set(g), key=repr)
    ts = sorted(set(t), key=repr)
    gi = dict((v, k) for k, v in enumerate(gs))
    ti = dict((v, k) for k, v in enumerate(ts))
    # residualise D on the two-way fixed effects by alternating
    # projections (the within transformation, iterated to convergence)
    r = list(D)
    for _ in range(500):
        ga = [0.0] * len(gs)
        gw = [0.0] * len(gs)
        for i in range(n):
            ga[gi[g[i]]] += r[i] * weights[i]
            gw[gi[g[i]]] += weights[i]
        for i in range(n):
            r[i] -= ga[gi[g[i]]] / gw[gi[g[i]]]
        ta = [0.0] * len(ts)
        tw = [0.0] * len(ts)
        for i in range(n):
            ta[ti[t[i]]] += r[i] * weights[i]
            tw[ti[t[i]]] += weights[i]
        for i in range(n):
            r[i] -= ta[ti[t[i]]] / tw[ti[t[i]]]
        if max(abs(sum(r[i] * weights[i] for i in range(n)
                       if g[i] == gg)) for gg in gs) < 1e-13:
            break
    denom = sum(weights[i] * D[i] * r[i] for i in range(n))
    if abs(denom) < 1e-14:
        raise ValueError("causdiddc: the treatment has no variation left "
                         "after the fixed effects; beta_fe is not "
                         "identified")
    cells = {}
    for i in range(n):
        if D[i] == 1.0:
            key = (g[i], t[i])
            cells[key] = cells.get(key, 0.0) + weights[i] * r[i] / denom
    return cells, r


def twfe(Y, D, group, period):
    r"""The two-way fixed effects coefficient and its decomposition."""
    Y, D, g, t, n = _panel(Y, D, group, period)
    w, resid = twfe_weights(D, g, t)
    denom = sum(D[i] * resid[i] for i in range(n))
    beta = sum(Y[i] * resid[i] for i in range(n)) / denom
    neg = dict((k, v) for k, v in w.items() if v < 0)
    return RichResult(payload={
        "estimate": beta,
        "beta_fe": beta,
        "weights": w,
        "n_negative": len(neg),
        "negative_mass": sum(abs(v) for v in neg.values()),
        "weight_sum": sum(w.values()),
        "n_treated_cells": len(w),
        "n": n,
        "method": ("two-way fixed effects (de Chaisemartin & "
                   "D'Haultfoeuille 2020, Theorem 1)"),
        "note": ("beta_fe is a weighted sum of cell effects whose "
                 "weights sum to 1 but may be negative; n_negative and "
                 "negative_mass say how much of the estimate runs "
                 "backwards. Compare against did_m"),
    })


def did_m(Y, D, group, period):
    r"""The paper's :math:`DID_M` estimator.

    For each pair of consecutive periods, compare the outcome change of
    cells that *switch* treatment against the change of cells whose
    treatment stays put, then average over switches, weighting by the
    number of switching observations. Unbiased for the average effect
    among switchers under common trends, with no restriction on
    heterogeneity.
    """
    Y, D, g, t, n = _panel(Y, D, group, period)
    cells = _cells(Y, D, g, t)
    periods = sorted(set(t), key=repr)
    num, den = 0.0, 0.0
    parts = []
    for k in range(1, len(periods)):
        t0, t1 = periods[k - 1], periods[k]
        stayers_up, stayers_dn = [], []
        for gg in set(g):
            a, b = cells.get((gg, t0)), cells.get((gg, t1))
            if a is None or b is None:
                continue
            if a[2] == b[2]:
                (stayers_up if a[2] == 1.0 else
                 stayers_dn).append((b[0] - a[0], b[1]))
        for gg in sorted(set(g), key=repr):
            a, b = cells.get((gg, t0)), cells.get((gg, t1))
            if a is None or b is None or a[2] == b[2]:
                continue
            ctrl = stayers_dn if b[2] == 1.0 else stayers_up
            if not ctrl:
                continue
            cw = sum(w for _, w in ctrl)
            trend = sum(d * w for d, w in ctrl) / cw
            eff = (b[0] - a[0]) - trend
            if b[2] == 0.0:              # a switch out of treatment
                eff = -eff
            num += eff * b[1]
            den += b[1]
            parts.append({"group": gg, "from": t0, "to": t1,
                          "effect": eff, "n": b[1],
                          "direction": "in" if b[2] == 1.0 else "out"})
    if den == 0:
        raise ValueError("causdiddc: no cell switches treatment between "
                         "consecutive periods, so DID_M is not defined")
    return RichResult(payload={
        "estimate": num / den,
        "did_m": num / den,
        "switches": parts,
        "n_switches": len(parts),
        "n_switching_obs": den,
        "n": n,
        "method": ("DID_M (de Chaisemartin & D'Haultfoeuille 2020): "
                   "switchers against stayers, period by period"),
        "note": ("unbiased for the average effect among switchers under "
                 "common trends, with no homogeneity assumption"),
    })


def causdiddc(Y, D, group, period):
    """Both estimators side by side, which is the paper's point."""
    fe = twfe(Y, D, group, period)
    try:
        dm = did_m(Y, D, group, period)
        dm_est = dm["estimate"]
        dm_n = dm["n_switches"]
    except ValueError:
        dm_est, dm_n = float("nan"), 0
    return RichResult(payload={
        "estimate": dm_est,
        "beta_fe": fe["beta_fe"],
        "did_m": dm_est,
        "weights": fe["weights"],
        "n_negative": fe["n_negative"],
        "negative_mass": fe["negative_mass"],
        "weight_sum": fe["weight_sum"],
        "n_switches": dm_n,
        "gap": fe["beta_fe"] - dm_est,
        "n": fe["n"],
        "method": ("TWFE against DID_M (de Chaisemartin & "
                   "D'Haultfoeuille 2020)"),
        "note": ("estimate is DID_M, the one that survives "
                 "heterogeneity; beta_fe is what a two-way fixed "
                 "effects regression would report, and gap is how far "
                 "apart they are"),
    })


def cheatsheet():
    return ("causdiddc: de Chaisemartin & D'Haultfoeuille (2020). The "
            "TWFE coefficient is sum w_gt Delta_gt over treated cells, "
            "weights from the residual of D on the two-way fixed "
            "effects; they sum to 1 and can be NEGATIVE, so beta_fe can "
            "have the opposite sign to every cell effect. DID_M compares "
            "switchers to stayers between consecutive periods and is "
            "unbiased under common trends whatever the heterogeneity.")

# public names resolved by fn/_lazy_map.json
causal_did_de_chaisemartin = twfe_weights
