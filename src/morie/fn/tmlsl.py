# morie.fn -- function file (rootcoder007/morie)
"""Super learner: convex combination of candidate predictors."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["superlrn", "tmle_super_learner"]


def superlrn(Z, Y, iters=500):
    """Weight candidate predictions by cross-validated risk, on the simplex.

    ``Z`` must hold CROSS-VALIDATED predictions -- each column the
    out-of-fold predictions of one candidate.  Feeding in-sample
    predictions instead is the way to make super learner pick the most
    overfit algorithm in the library every time, and nothing in the
    arithmetic can detect it.

    The weights are restricted to the SIMPLEX, which is what bounds the
    ensemble whenever the candidates are bounded and is what the oracle
    result requires.  The discrete super learner -- the single best
    candidate -- is the best vertex of that simplex and is returned
    beside the ensemble, because the ensemble is only worth its
    complexity if it beats it.

    Optimisation is Frank-Wolfe with a FIXED step schedule
    2/(t + 2) and a fixed budget, so both language arms return
    identical weights; ties in the vertex choice break on the lowest
    index.

    Formula: Psi_SL(W) = sum_k alpha_k Psi_k(W),
             alphahat = argmin_{alpha in simplex} sum_i (Y_i - (Z alpha)_i)^2

    Parameters
    ----------
    Z : array-like, shape (n, K)
        Cross-validated predictions, one column per candidate.
    Y : array-like
        Outcome.
    iters : int
        Frank-Wolfe steps (fixed budget).

    Returns
    -------
    RichResult
        ``weights``, ``risk`` (per candidate CV risk), ``sl_risk``,
        ``discrete_risk``, ``discrete_index`` (one-based),
        ``fitted``, ``beats_discrete`` (1 when the ensemble wins),
        ``n``, ``K``.

    References
    ----------
    van der Laan, Polley & Hubbard (2007), Super Learner, UC Berkeley
    Division of Biostatistics Working Paper 222, also Statistical
    Applications in Genetics and Molecular Biology 6(1) -- the row's
    own citation.  The working paper PDF could not be downloaded
    (the bepress endpoint returned an empty body); the construction is
    taken from Polley's dissertation, Super Learner, UC Berkeley
    (escholarship qt4qn0067v), which was fetched in full and states it
    as "combine alphahat with Psihat_k(W) ... to create the final super
    learner fit Psihat_SL(W) = sum_k alphahat_k Psihat_k(W)", with
    alpha restricted to the convex combination and the nodes of the
    convex hull corresponding to "the usual cross-validation selector
    we refer to as the discrete super learner".
    """
    Z = C.mat(Z)
    n = len(Z)
    if n < 2:
        raise ValueError("at least two observations are required")
    K = len(Z[0])
    if any(len(r) != K for r in Z):
        raise ValueError("every row must score every candidate")
    Y = C.vec(Y)
    if len(Y) != n:
        raise ValueError("Y must have one entry per observation")
    if K < 1:
        raise ValueError("at least one candidate is required")
    risk = [sum((Y[i] - Z[i][k]) ** 2 for i in range(n)) / n
            for k in range(K)]
    dbest = min(range(K), key=lambda k: (risk[k], k))
    a = [0.0] * K
    a[dbest] = 1.0
    for t in range(int(iters)):
        f = [sum(Z[i][k] * a[k] for k in range(K)) for i in range(n)]
        gr = [-2.0 / n * sum((Y[i] - f[i]) * Z[i][k] for i in range(n))
              for k in range(K)]
        v = min(range(K), key=lambda k: (gr[k], k))
        g = 2.0 / (t + 2.0)
        a = [(1.0 - g) * a[k] + (g if k == v else 0.0) for k in range(K)]
    s = sum(a)
    a = [v / s for v in a]
    fit = [sum(Z[i][k] * a[k] for k in range(K)) for i in range(n)]
    slr = sum((Y[i] - fit[i]) ** 2 for i in range(n)) / n
    return RichResult(payload={
        "weights": a, "risk": risk, "sl_risk": slr,
        "discrete_risk": risk[dbest], "discrete_index": float(dbest + 1),
        "fitted": fit,
        "beats_discrete": 1.0 if slr <= risk[dbest] else 0.0,
        "n": float(n), "K": float(K),
        "method": "Super learner: simplex-constrained ensemble of candidates"})


tmle_super_learner = superlrn


def cheatsheet():
    return "tmlsl: alpha on the simplex minimising CV risk; discrete SL = best vertex"
