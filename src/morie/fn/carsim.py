# morie.fn -- function file (rootcoder007/morie)
"""CAR conditional simulation."""

import math

from ._richresult import RichResult
from ._rng import random_normal


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def _chol(Q):
    n = len(Q)
    L = [[0.0] * n for _ in range(n)]
    for j in range(n):
        s = Q[j][j]
        for k in range(j):
            s -= L[j][k] * L[j][k]
        if s <= 0.0:
            raise ValueError("I - rho W is not positive definite for this rho")
        L[j][j] = math.sqrt(s)
        for i in range(j + 1, n):
            t = Q[i][j]
            for k in range(j):
                t -= L[i][k] * L[j][k]
            L[i][j] = t / L[j][j]
    return L


def carsim(W, rho, sigma2, nsim=9, seed=0):
    r"""Simulate a zero-mean Gaussian CAR field ``x ~ N(0, sigma2 (I - rho W)^{-1})`` from its precision.

    With ``Q = (I - rho W) / sigma2 = L L'`` (Cholesky, ``W`` symmetric) each
    draw is ``x = L'^{-1} z``, ``z ~ N(0, I)``, which has covariance ``Q^{-1}``
    (Rue and Held 2005, Algorithm 2.4). Draw ``k`` uses Philox stream ``k`` of
    ``seed`` so the R arm reproduces the same fields.

    References
    ----------
    Rue, H. and Held, L. (2005). *Gaussian Markov Random Fields: Theory and
    Applications*. Chapman and Hall/CRC.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> r = carsim(W, 0.4, 2.0, nsim=2)
    >>> len(r["draws"]), len(r["draws"][0])
    (2, 3)
    """
    Wm = _mat(W)
    n = len(Wm)
    if any(Wm[i][j] != Wm[j][i] for i in range(n) for j in range(n)):
        raise ValueError("W must be symmetric")
    Q = [[((1.0 if i == j else 0.0) - float(rho) * Wm[i][j]) / float(sigma2) for j in range(n)] for i in range(n)]
    L = _chol(Q)
    draws = []
    for k in range(int(nsim)):
        z = [float(v) for v in random_normal(n, seed=seed, stream=k)]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = z[i]
            for j in range(i + 1, n):
                s -= L[j][i] * x[j]
            x[i] = s / L[i][i]
        draws.append(x)
    return RichResult(payload={"draws": draws, "nsim": int(nsim), "seed": int(seed)})


carsim_fn = carsim


def cheatsheet() -> str:
    return "carsim(W, rho, sigma2, nsim=9, seed=0) -> CAR draws x = L'^{-1} z from Q = (I - rho W)/sigma2 = LL'."
