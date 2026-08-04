# morie.fn -- function file (rootcoder007/morie)
"""Map standard normal draws to additive logistic normal compositions."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lognormdraw', 'logistic_normal_sample']


def lognormdraw(Z, mu, Sigma, ref=None, total=1.0):
    """Map standard normal draws to additive logistic normal compositions.

    Formula: x_r = alr^-1( mu + L z_r ),  Sigma = L L' the lower Cholesky factor

    Parameters
    ----------
    Z : array-like, shape (n, D-1)
        Standard normal draws supplied by the caller, one row per composition.  The noise is an argument, not drawn internally, so the function is deterministic.
    mu : array-like
        Mean of the additive log-ratio coordinates, length D - 1.
    Sigma : array-like, shape (D-1, D-1)
        Covariance of the additive log-ratio coordinates; must be positive definite.
    ref : int
        1-based index the reference part is restored to; the default is the last position D.
    total : float
        Constant kappa each returned composition sums to.

    Returns
    -------
    RichResult
        ``compositions``, ``alr``, ``L``, ``ref``, ``n``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The log-ratio algebra and the additive logistic normal law were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sects. 4.1 and 4.3, which attribute the law to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sampling the additive logistic normal is sampling the multivariate normal in alr coordinates and inverting the transform, since the law is defined by exactly that construction.  The caller supplies the standard normal matrix Z so that the result is reproducible and identical in both language arms; no random number generator is touched here.
    """
    Zm = C.mat(Z)
    n = len(Zm)
    if n == 0:
        raise ValueError("Z must have at least one row")
    p = len(Zm[0])
    D = p + 1
    mu = C.vec(mu)
    if len(mu) != p:
        raise ValueError("mu must have one entry per column of Z")
    Sg = C.mat(Sigma)
    if len(Sg) != p or len(Sg[0]) != p:
        raise ValueError("Sigma must match the number of columns of Z")
    L = C.chol(Sg)
    k = D if ref is None else int(ref)
    if not 1 <= k <= D:
        raise ValueError("ref must be a 1-based part index")
    idx = [i for i in range(1, D + 1) if i != k]
    t = float(total)
    Y, out = [], []
    for r in range(n):
        y = [mu[a] + sum(L[a][b] * Zm[r][b] for b in range(a + 1)) for a in range(p)]
        Y.append(y)
        full = [0.0] * D
        for pos, i in enumerate(idx):
            full[i - 1] = y[pos]
        m = max(full)
        e = [math.exp(val - m) for val in full]
        s = sum(e)
        out.append([t * val / s for val in e])
    return RichResult(payload={
        "compositions": out, "alr": Y, "L": L, "ref": k, "n": n, "D": D,
        "method": "Additive logistic normal draws from supplied noise"})


logistic_normal_sample = lognormdraw


def cheatsheet():
    return 'aitlns: Map standard normal draws to additive logistic normal compositions.'
