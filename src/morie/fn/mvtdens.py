"""Multivariate t density and draws.

Kotz, S. & Nadarajah, S. (2004). Multivariate t Distributions and Their Applications. Cambridge, ch. 1.
"""

import math

from ._mvcore import as_matrix, cholesky, forward, logdet_chol, points
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._rrng_core import qchisq

__all__ = ["mvtdens"]


def mvtdens(x=None, loc=(0.0, 0.0), scale=((1.0, 0.0), (0.0, 1.0)), df=1.0, n=0, seed=0):
    r"""log f(x) = log Gamma((nu + d)/2) - log Gamma(nu/2) - (d/2) log(nu pi) - (1/2) log|Sigma|
    - ((nu + d)/2) log(1 + (x - mu)' Sigma^{-1} (x - mu)/nu).

    Draws are mu + L z / sqrt(W/nu) with z Philox normals (stream 0) and W a
    chi-square by inversion of Philox uniforms (stream 1).

    Parameters
    ----------
    x : one point or list of points, optional
    loc : sequence of d floats
    scale : d x d positive-definite scale matrix Sigma
    df : float
        Degrees of freedom nu > 0.
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, random.

    References
    ----------
    Kotz, S. & Nadarajah, S. (2004). Multivariate t Distributions and Their Applications, ch. 1.
    Matches ``mvtnorm::dmvt(x, delta = loc, sigma = scale, df = df, log = FALSE)``.

    Examples
    --------
    >>> round(mvtdens([0.0, 0.0], df=1.0)["pdf"] * 2 * math.pi, 12)
    1.0
    """
    mu = [float(v) for v in loc]
    S = as_matrix(scale)
    d = len(mu)
    if len(S) != d or not df > 0:
        raise ValueError("loc and scale dimensions must agree and df > 0")
    L = cholesky(S)
    ld = logdet_chol(L)
    c = math.lgamma((df + d) / 2) - math.lgamma(df / 2) - d / 2 * math.log(df * math.pi) - 0.5 * ld
    payload = {}
    if x is not None:
        pts, single = points(x)
        lp = []
        for pt in pts:
            z = forward(L, [a - b for a, b in zip(pt, mu)])
            lp.append(c - (df + d) / 2 * math.log1p(sum(v * v for v in z) / df))
        payload["logpdf"] = lp[0] if single else lp
        payload["pdf"] = math.exp(lp[0]) if single else [math.exp(v) for v in lp]
    if n:
        zs = random_normal(int(n) * d, seed=seed, stream=0)
        ws = [qchisq(u, df) for u in random_uniform(int(n), seed=seed, stream=1)]
        payload["random"] = [
            [mu[i] + sum(L[i][k] * zs[r * d + k] for k in range(i + 1)) / math.sqrt(ws[r] / df) for i in range(d)]
            for r in range(int(n))
        ]
    return RichResult(title="Multivariate t", summary_lines=[("dimension", d), ("df", df)], payload=payload)


def cheatsheet():
    return "mvtdens: multivariate t density, Philox draws."
