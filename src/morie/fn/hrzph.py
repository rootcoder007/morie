# morie.fn -- function file (rootcoder007/morie)
"""Proportional hazards by partial likelihood, with the Tsiatis baseline.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 6.2.1, pages 201-203 (volume
[Pages 189-232], read as rendered page images).  Note the sign
convention: the book writes the model as log Lambda_0(Y) = X beta + U
with U extreme-value (6.31), so the conditional hazard carries
exp(-x beta), not exp(+x beta), and every formula below follows that.
The partial likelihood (6.32) is

    L_np(b) = prod_i [ exp(-X_i b) / sum_{j in R(Y_i)} exp(-X_j b) ],

R(y) = {i : Y_i >= y} the risk set at y, maximised here by
Newton-Raphson on its logarithm.  The integrated baseline hazard is
Tsiatis (1981) sample analogue of (6.35), printed as (6.36):

    Lambda_n0(y) = sum_{i : Y_i <= y} 1 / sum_{j in R(Y_i)} exp(-X_j b_n).

The covariance of b_n is the inverse observed information, V_nb of
(6.34).  A coefficient here is therefore the NEGATIVE of the usual Cox
coefficient for the same data; that is the book convention, not an
error, and the payload says so.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_proportional_hazards"]


def horowitz_proportional_hazards(t, x, event=None, max_iter=100, tol=1e-12):
    """Partial-likelihood beta and the Tsiatis integrated baseline hazard.

    Parameters
    ----------
    t : array-like
        Observed durations Y.
    x : array-like
        n-by-p covariate matrix WITHOUT an intercept: the baseline
        absorbs it, exactly as in the Cox model.
    event : array-like, optional
        1 if the duration ended in the event, 0 if right censored.
        All ones if omitted.
    max_iter, tol : Newton controls.

    Returns
    -------
    beta_hat : the maximiser of (6.32), in the books exp(-x beta) sign
    h0_hat   : the jumps of Lambda_n0 at the event times
    Lambda0  : the cumulative baseline at those times
    """
    tt = core.vec(t)
    XX = core.mat(x)
    n = len(tt)
    if n == 0:
        raise ValueError("horowitz_proportional_hazards: t is empty")
    if len(XX) != n:
        raise ValueError("horowitz_proportional_hazards: x has a different number of rows than t")
    p = len(XX[0])
    if event is None:
        ev = [1.0] * n
    else:
        ev = core.vec(event)
        if len(ev) != n:
            raise ValueError("horowitz_proportional_hazards: event has a different length than t")
    order = sorted(range(n), key=lambda i: (tt[i], i))
    beta = [0.0] * p
    for _ in range(int(max_iter)):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for i in order:
            if ev[i] == 0.0:
                continue
            s0 = 0.0
            s1 = [0.0] * p
            s2 = [[0.0] * p for _ in range(p)]
            for j in range(n):
                if tt[j] < tt[i]:
                    continue
                e = 0.0
                for k in range(p):
                    e += XX[j][k] * beta[k]
                w = math.exp(min(max(-e, -300.0), 300.0))
                s0 += w
                for a in range(p):
                    s1[a] += w * XX[j][a]
                    for b in range(p):
                        s2[a][b] += w * XX[j][a] * XX[j][b]
            for a in range(p):
                g[a] += -XX[i][a] + s1[a] / s0
                for b in range(p):
                    H[a][b] += -(s2[a][b] / s0 - (s1[a] / s0) * (s1[b] / s0))
        step = core.ridgesolve(H, g, 1e-12)
        d = 0.0
        for a in range(p):
            beta[a] -= step[a]
            if abs(step[a]) > d:
                d = abs(step[a])
        if d < tol:
            break
    # Tsiatis (6.36): the jumps of Lambda_n0 at the event times
    times = []
    jumps = []
    cum = []
    running = 0.0
    for i in order:
        if ev[i] == 0.0:
            continue
        s0 = 0.0
        for j in range(n):
            if tt[j] < tt[i]:
                continue
            e = 0.0
            for k in range(p):
                e += XX[j][k] * beta[k]
            s0 += math.exp(min(max(-e, -300.0), 300.0))
        running += 1.0 / s0
        times.append(tt[i])
        jumps.append(1.0 / s0)
        cum.append(running)
    # V_nb of (6.34): minus the inverse observed information
    se = []
    for a in range(p):
        e = [0.0] * p
        e[a] = 1.0
        col = core.ridgesolve([[-H[r][c] for c in range(p)] for r in range(p)], e, 1e-12)
        se.append(math.sqrt(col[a]) if col[a] > 0.0 else float("nan"))
    return RichResult(
        title="Proportional hazards, partial likelihood",
        summary_lines=[("n", n), ("events", len(times))],
        payload={
            "estimate": beta[0],
            "beta_hat": beta,
            "se": se,
            "h0_hat": jumps,
            "event_times": times,
            "Lambda0": cum,
            "n": n,
            "n_events": len(times),
            "method": "Horowitz (2009) eq. (6.32) partial likelihood with exp(-x b); baseline by (6.36)",
        },
    )


def cheatsheet():
    return "hrzph: Proportional hazards model with nonparametric baseline and parametric F"
