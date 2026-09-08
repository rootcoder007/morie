# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Bayesian optimisation: expected improvement under a GP posterior.

The stub carried the label "Wang-Frazier (2017)".  Nothing by that
author pair in that year matches an acquisition function of this
description in Crossref; the attribution is recorded as UNVERIFIED.
The acquisition implemented is the one the stub's formula describes --
maximise an acquisition under the GP posterior -- in its standard
form, expected improvement, from Jones, Schonlau and Welch (1998),
"Efficient global optimization of expensive black-box functions",
Journal of Global Optimization 13(4):455-492,
doi:10.1023/A:1008306431147, equation (15).  For minimisation, with
f_min the best value seen so far,

    EI(x) = (f_min - mu(x)) Phi(z) + sigma(x) phi(z),
    z     = (f_min - mu(x)) / sigma(x),     EI = 0 when sigma(x) = 0.

The exact vanishing at zero posterior variance is what stops the
search from re-evaluating a point it already knows, and it is the
anchor the tests use.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_variational_bayes_opt"]

_SQ2PI = math.sqrt(2.0 * math.pi)


def _k(A, B, ell, var):
    out = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            s = 0.0
            for c in range(len(A[i])):
                d = A[i][c] - B[j][c]
                s += d * d
            row.append(var * math.exp(-0.5 * s / (ell * ell)))
        out.append(row)
    return out


def gp_variational_bayes_opt(X, y, X_grid, lengthscale=1.0, variance=1.0, noise=1e-6, xi=0.0):
    """Expected improvement over a candidate grid, and the point that maximises it."""
    A = core.mat(X)
    n = len(A)
    if n == 0:
        raise ValueError("gp_variational_bayes_opt: X is empty")
    yv = core.vec(y)
    if len(yv) != n:
        raise ValueError("gp_variational_bayes_opt: X and y have different lengths")
    G = core.mat(X_grid)
    if len(G) == 0:
        raise ValueError("gp_variational_bayes_opt: the candidate grid is empty")
    if len(G[0]) != len(A[0]):
        raise ValueError("gp_variational_bayes_opt: grid and X have different dimensions")
    ell = float(lengthscale)
    var = float(variance)
    s2 = float(noise)
    if ell <= 0 or var <= 0:
        raise ValueError("gp_variational_bayes_opt: lengthscale and variance must be positive")
    if s2 < 0:
        raise ValueError("gp_variational_bayes_opt: noise must be non-negative")
    K = [[_k([A[i]], [A[j]], ell, var)[0][0] + (s2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    al = core.cholsolve(K, yv)
    Ks = _k(G, A, ell, var)
    fmin = min(yv)
    mu = []
    sd = []
    ei = []
    for j in range(len(G)):
        m = sum(Ks[j][i] * al[i] for i in range(n))
        v = core.cholsolve(K, Ks[j])
        q = max(var - sum(Ks[j][i] * v[i] for i in range(n)), 0.0)
        s = math.sqrt(q)
        mu.append(m)
        sd.append(s)
        if s <= 0.0:
            ei.append(0.0)
        else:
            imp = fmin - m - float(xi)
            z = imp / s
            ei.append(imp * core.pnorm(z) + s * math.exp(-0.5 * z * z) / _SQ2PI)
    best = 0
    for j in range(len(ei)):
        if ei[j] > ei[best]:
            best = j
    return RichResult(
        title="Bayesian optimisation (expected improvement)",
        summary_lines=[("n", n), ("grid", len(G)), ("f_min", fmin)],
        payload={
            "estimate": ei[best],
            "acquisition": ei,
            "mean": mu,
            "sd": sd,
            "next_index": best + 1,
            "next_point": list(G[best]),
            "f_min": fmin,
            "n": n,
            "method": "expected improvement, Jones, Schonlau & Welch (1998) eq. (15); stub attribution 'Wang-Frazier (2017)' UNVERIFIED",
        },
    )


def cheatsheet():
    return "gpvbo: Bayesian optimisation by expected improvement"
