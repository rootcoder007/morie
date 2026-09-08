# morie.fn -- function file (rootcoder007/morie)
"""Propensity truncation and its effect on a targeted estimate."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmlestab", "tmle_stabilized"]


def tmlestab(Y, A, QAW, Q1W, Q0W, g1W, gbounds=None):
    """Stabilize a targeted estimate by truncating the propensity score.

    Near-positivity violations are the failure mode that matters: a
    single observation with g(W) = 0.001 gets a clever-covariate weight
    of 1000 and can move the whole estimate on its own.  Truncation
    caps that weight at the price of bias, and this returns the whole
    trade-off curve rather than one point, so the choice of bound is
    made on evidence.

    ``max_weight`` at each bound is the diagnostic to read: if it is
    far above n^(1/2) the estimate is effectively resting on a handful
    of observations.

    Formula: g_trunc = min(max(g, delta), 1 - delta);
             re-target at each delta and report psi, se and max weight

    Parameters
    ----------
    Y, A : array-like
        Outcome in [0, 1] and binary treatment.
    QAW, Q1W, Q0W : array-like
        Initial outcome predictions.
    g1W : array-like
        Initial propensity.
    gbounds : sequence of float, optional
        Truncation levels to try (default: 0.001, 0.01, 0.025, 0.05,
        0.10).

    Returns
    -------
    RichResult
        ``gbounds``, ``estimate`` (one per bound), ``se``,
        ``max_weight``, ``n_truncated``, ``spread`` (range of the
        estimates), ``n``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan), which bounds the propensity
    with ``.bound(g1W, gbounds)`` before forming the clever covariates.
    The stabilizing role of that bound under near-positivity violations
    is the subject of Gruber & van der Laan (2010), A targeted maximum
    likelihood estimator of a causal effect on a bounded continuous
    outcome, International Journal of Biostatistics 6(1), Article 26,
    which this row cites.
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
    gb = [0.001, 0.01, 0.025, 0.05, 0.10] if gbounds is None \
        else C.vec(gbounds)
    if any(not 0.0 < v < 0.5 for v in gb):
        raise ValueError("each bound must lie strictly between 0 and 0.5")
    est = []
    ses = []
    mw = []
    nt = []
    for d in gb:
        fit = T.target(Y, A, QAW, Q1W, Q0W, g1W, d)
        mu1, mu0, ic1, ic0 = T.curves(Y, A, fit)
        ic = [ic1[i] - ic0[i] for i in range(n)]
        est.append(mu1 - mu0)
        ses.append(math.sqrt(C.var(ic, 1) / n))
        mw.append(max(max(fit["H1"]), max(fit["H0"])))
        nt.append(float(sum(1 for v in g1W if v < d or v > 1.0 - d)))
    return RichResult(payload={
        "gbounds": gb, "estimate": est, "se": ses, "max_weight": mw,
        "n_truncated": nt, "spread": max(est) - min(est), "n": float(n),
        "method": "Propensity truncation trade-off for a targeted estimate"})


tmle_stabilized = tmlestab


def cheatsheet():
    return "tmlsta: re-target across truncation bounds; read max_weight"
