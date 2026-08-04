# morie.fn -- function file (rootcoder007/morie)
"""Quasi-score residual before and after targeting."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmleqs", "tmle_quasi_score"]


def tmleqs(Y, A, QAW, Q1W, Q0W, g1W, gbound=0.025):
    """How far the initial fit was from solving the efficient score equation.

    This is the diagnostic that says whether targeting DID anything.
    The initial fit generally leaves a non-zero empirical score; the
    targeted fit must drive it to numerical zero, and the ratio of the
    two is the honest measure of how much the estimate moved and why.

    ``score_init`` scaled by sqrt(n)/sd is the more interpretable
    figure: it is roughly the number of standard errors of bias the
    plug-in estimate carried.

    Formula: D*(psi)(O) = (A/g1 - (1-A)/g0)(Y - Q(A,W))
                          + Q(1,W) - Q(0,W) - psi;
             targeting solves (1/n) sum D*(psi)(O_i) = 0

    Parameters
    ----------
    Y, A : array-like
        Outcome in [0, 1] and binary treatment.
    QAW, Q1W, Q0W : array-like
        Initial outcome predictions.
    g1W : array-like
        Initial propensity.
    gbound : float
        Propensity truncation level.

    Returns
    -------
    RichResult
        ``score_init``, ``score_final``, ``score_init_scaled``,
        ``reduction``, ``psi_init``, ``psi_final``, ``shift``,
        ``epsilon``, ``n``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan): the clever covariates
    H1W = A/g1W and H0W = (1-A)/g0W and the influence curve assembled
    in ``calcParameters`` are the components of D* used here.  The
    score-equation view of the targeting step is van der Laan &
    Rubin (2006), Targeted maximum likelihood learning, International
    Journal of Biostatistics 2(1), Article 11.
    """
    Y = C.vec(Y)
    A = C.vec(A)
    n = len(Y)
    QAW = C.vec(QAW)
    Q1W = C.vec(Q1W)
    Q0W = C.vec(Q0W)
    g1W = C.vec(g1W)
    if any(len(v) != n for v in (A, QAW, Q1W, Q0W, g1W)):
        raise ValueError("every argument must have one entry per observation")
    if n < 2:
        raise ValueError("at least two observations are required")
    g1 = [T.bound(v, gbound, 1.0 - gbound) for v in g1W]
    g0 = [1.0 - v for v in g1]
    psi0 = sum(Q1W) / n - sum(Q0W) / n
    d0 = [(A[i] / g1[i] - (1.0 - A[i]) / g0[i]) * (Y[i] - QAW[i])
          + Q1W[i] - Q0W[i] - psi0 for i in range(n)]
    s0 = sum(d0) / n
    fit = T.target(Y, A, QAW, Q1W, Q0W, g1W, gbound)
    mu1, mu0, ic1, ic0 = T.curves(Y, A, fit)
    psi1 = mu1 - mu0
    d1 = [(A[i] / g1[i] - (1.0 - A[i]) / g0[i])
          * (Y[i] - fit["QAstar"][i])
          + fit["Q1star"][i] - fit["Q0star"][i] - psi1 for i in range(n)]
    s1 = sum(d1) / n
    sd0 = C.sd(d0, 1)
    return RichResult(payload={
        "score_init": s0, "score_final": s1,
        "score_init_scaled": s0 * math.sqrt(n) / sd0 if sd0 > 0 else float("nan"),
        "reduction": abs(s1) / abs(s0) if s0 != 0.0 else float("nan"),
        "psi_init": psi0, "psi_final": psi1, "shift": psi1 - psi0,
        "epsilon": fit["epsilon"], "n": float(n),
        "method": "Efficient-score residual before and after targeting"})


tmle_quasi_score = tmleqs


def cheatsheet():
    return "tmlqsa: mean D* before vs after the fluctuation; after must be ~0"
