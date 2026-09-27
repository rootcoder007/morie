"""Multivariate normal density and draws.

Anderson, T. W. (2003). An Introduction to Multivariate Statistical Analysis, 3rd ed. Wiley, ch. 2.
"""

import math

from ._mvcore import as_matrix, cholesky, forward, logdet_chol, points
from ._richresult import RichResult
from ._rng import random_normal

__all__ = ["mvnormdens"]


def mvnormdens(x=None, mean=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0)), n=0, seed=0):
    r"""log f(x) = -(d/2) log(2 pi) - (1/2) log|Sigma| - (1/2)(x - mu)' Sigma^{-1} (x - mu).

    Computed through the Cholesky factor Sigma = L L' (z = L^{-1}(x - mu)); draws
    are mu + L z with z from morie's Philox normals (identical in the R arm).

    Parameters
    ----------
    x : sequence (one point) or list of points, optional
    mean : sequence of d floats
    cov : d x d positive-definite matrix
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, mahalanobis (squared), random (n x d).

    References
    ----------
    Anderson, T. W. (2003). An Introduction to Multivariate Statistical Analysis, 3rd ed., ch. 2.
    Matches ``mvtnorm::dmvnorm(x, mean, sigma)``.

    Examples
    --------
    >>> round(mvnormdens([0.0, 0.0])["pdf"] * 2 * math.pi, 12)
    1.0
    """
    mu = [float(v) for v in mean]
    S = as_matrix(cov)
    d = len(mu)
    if len(S) != d:
        raise ValueError("mean and cov dimensions differ")
    L = cholesky(S)
    ld = logdet_chol(L)
    payload = {}
    if x is not None:
        pts, single = points(x)
        m2 = []
        for pt in pts:
            if len(pt) != d:
                raise ValueError("points must have the dimension of mean")
            z = forward(L, [a - b for a, b in zip(pt, mu)])
            m2.append(sum(v * v for v in z))
        lp = [-0.5 * d * math.log(2 * math.pi) - 0.5 * ld - 0.5 * q for q in m2]
        payload["logpdf"] = lp[0] if single else lp
        payload["pdf"] = math.exp(lp[0]) if single else [math.exp(v) for v in lp]
        payload["mahalanobis"] = m2[0] if single else m2
    if n:
        zs = random_normal(int(n) * d, seed=seed, stream=0)
        payload["random"] = [
            [mu[i] + sum(L[i][k] * zs[r * d + k] for k in range(i + 1)) for i in range(d)] for r in range(int(n))
        ]
    return RichResult(title="Multivariate normal", summary_lines=[("dimension", d)], payload=payload)


def cheatsheet():
    return "mvnormdens: multivariate normal density via Cholesky, Philox draws."
