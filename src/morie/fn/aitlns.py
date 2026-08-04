# morie.fn -- function file (rootcoder007/morie)
"""Sampling from the additive logistic-normal."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lgtnsim", "logistic_normal_sample"]


def lgtnsim(mu, Sigma, n, seed=1, total=1.0):
    """Draw compositions from a logistic-normal via the alr inverse.

    Sampling on the simplex is done the only sane way: sample in the
    unconstrained coordinates and map back.  Trying to sample the
    parts directly and renormalise gives a different distribution
    entirely -- the closure is not measure-preserving.

    The Cholesky factor is used rather than a symmetric square root
    because it is unique for a positive-definite Sigma, so the two
    language arms map the same normal draws to the same compositions.
    The draws themselves come from the shared pinned Lehmer stream.

    Formula: Z ~ N(0, I_{D-1});  Y = mu + L Z with L L' = Sigma;
             X = alr^-1(Y) = C(exp(Y_1), ..., exp(Y_{D-1}), 1)

    Parameters
    ----------
    mu : array-like
        Mean of the alr coordinates, length D-1.
    Sigma : array-like, shape (D-1, D-1)
        Positive-definite covariance of the alr coordinates.
    n : int
        Number of compositions drawn.
    seed : int
        Seed for the pinned generator.
    total : float
        Constant each composition sums to.

    Returns
    -------
    RichResult
        ``sample`` (n x D), ``alr`` (n x (D-1)), ``center``,
        ``mean_alr``, ``n``, ``D``.

    References
    ----------
    Aitchison (1986), The Statistical Analysis of Compositional Data,
    Chapter 6: a composition is logistic-normal exactly when its
    additive log-ratio vector is multivariate normal, so a draw is
    alr^-1 of a normal draw.  The reference part is the LAST, matching
    the sibling modules ``aitalr`` and ``aitalri``.
    """
    mu = C.vec(mu)
    p = len(mu)
    if p < 1:
        raise ValueError("mu must have at least one entry")
    S = C.mat(Sigma)
    if len(S) != p or any(len(r) != p for r in S):
        raise ValueError("Sigma must be (D-1) x (D-1)")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    L = C.chol(S)
    g = C.Lcg(seed)
    D = p + 1
    k = float(total)
    Y = []
    Xs = []
    for _ in range(n):
        z = [g.norm() for _ in range(p)]
        y = [mu[i] + sum(L[i][j] * z[j] for j in range(i + 1))
             for i in range(p)]
        Y.append(y)
        e = [math.exp(v) for v in y] + [1.0]
        s = sum(e)
        Xs.append([k * v / s for v in e])
    ym = [sum(Y[t][i] for t in range(n)) / n for i in range(p)]
    e = [math.exp(v) for v in mu] + [1.0]
    s = sum(e)
    return RichResult(payload={
        "sample": Xs, "alr": Y, "center": [k * v / s for v in e],
        "mean_alr": ym, "n": float(n), "D": float(D),
        "method": "Logistic-normal sampling via alr^-1"})


logistic_normal_sample = lgtnsim


def cheatsheet():
    return "aitlns: Y = mu + L Z, X = alr^-1(Y); Cholesky L, pinned stream"
