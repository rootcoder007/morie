# morie.fn -- function file (rootcoder007/morie)
"""Network meta-analysis by a linear mixed model on contrasts."""

import math

from . import _macore as ma
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_network_lme"]


def ma_network_lme(yi, vi, design):
    """Borrow strength across a whole network of pairwise comparisons.

    Two treatments never compared head to head still have a comparison
    implied by the trials each has against a common third: the consistency
    assumption turns a set of unconnected pairwise meta-analyses into one
    model with ``T - 1`` free parameters.  What is bought is a comparison
    that no trial made; what is risked is that the assumption is false,
    which is why the residual heterogeneity is reported alongside.

    Formula: ``y_i = d_{t2(i)} - d_{t1(i)} + u_i + e_i`` with ``d`` of the
    reference treatment fixed at zero, ``Var(u) = tau^2`` common across
    contrasts, weights ``1/(v_i + tau^2)`` -- Salanti et al. (2008)
    Section 3.  ``tau^2`` is the moment estimator from the residual Q.

    Parameters
    ----------
    yi : array-like, shape (n,)
        Contrast estimates: the effect of the comparator against the
        baseline of the same row of ``design``.
    vi : array-like, shape (n,)
        Their sampling variances, strictly positive.
    design : array-like, shape (n, 2)
        Baseline and comparator treatment labels per study.

    Returns
    -------
    RichResult
        ``theta`` (effects of every treatment against the reference, the
        reference itself included as zero), ``se_theta``, ``ranks``
        (1 = smallest effect), ``tau2``, ``QE``, ``treatments``, ``n``,
        ``T``.

    References
    ----------
    Salanti, G., Higgins, J. P. T., Ades, A. E. and Ioannidis, J. P. A.
    (2008).  Evaluation of networks of randomized trials.  Statistical
    Methods in Medical Research 17(3):279-301.
    doi:10.1177/0962280207080643.
    """
    y = [float(t) for t in core.vec(yi)]
    v = [float(t) for t in core.vec(vi)]
    n = len(y)
    if n == 0:
        raise ValueError("no studies")
    if len(v) != n:
        raise ValueError("yi and vi must have equal length")
    if any(t <= 0.0 for t in v):
        raise ValueError("sampling variances must be strictly positive")
    X, treats, T = ma.net_design(design)
    if len(X) != n:
        raise ValueError("design must have one row per study")
    p = T - 1
    w0 = [1.0 / t for t in v]
    b0, _, _ = ma.wls(X, y, w0)
    resid = [y[i] - sum(X[i][r] * b0[r] for r in range(p)) for i in range(n)]
    QE = sum(w0[i] * resid[i] * resid[i] for i in range(n))
    df = n - p
    denom = 0.0
    if df > 0:
        A = [[sum(w0[i] * X[i][r] * X[i][s] for i in range(n))
              for s in range(p)] for r in range(p)]
        A2 = [[sum(w0[i] * w0[i] * X[i][r] * X[i][s] for i in range(n))
               for s in range(p)] for r in range(p)]
        inv = []
        for j in range(p):
            e = [1.0 if r == j else 0.0 for r in range(p)]
            inv.append(core.ridgesolve(A, e, 1e-12))
        tr = sum(sum(inv[j][r] * A2[r][j] for r in range(p)) for j in range(p))
        denom = sum(w0) - tr
    tau2 = 0.0
    if denom > 0.0:
        tau2 = (QE - df) / denom
        if tau2 < 0.0:
            tau2 = 0.0
    w = [1.0 / (v[i] + tau2) for i in range(n)]
    beta, cov, _ = ma.wls(X, y, w)
    theta = [0.0] + list(beta)
    se = [0.0] + [math.sqrt(cov[j][j]) if cov[j][j] > 0.0 else float("nan")
                  for j in range(p)]
    order = sorted(range(T), key=lambda j: (theta[j], j))
    ranks = [0] * T
    for r, j in enumerate(order):
        ranks[j] = r + 1
    return RichResult(payload={
        "theta": theta, "se_theta": se, "ranks": ranks, "tau2": tau2,
        "QE": QE, "treatments": treats, "n": n, "T": T,
        "method": "Network meta-analysis on contrasts"})


def cheatsheet():
    return "manlmm: network meta-analysis by a mixed model on contrasts"


# compact alias per ledger/NAMING.md
manetworklme = ma_network_lme
