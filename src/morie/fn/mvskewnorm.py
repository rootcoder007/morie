"""Multivariate skew-normal density (Azzalini & Dalla Valle).

Azzalini, A. & Dalla Valle, A. (1996). The multivariate skew-normal distribution. Biometrika 83, 715-726.
"""

import math

from ._mvcore import as_matrix, cholesky, forward, logdet_chol, points
from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = ["mvskewnorm"]


def mvskewnorm(x, xi=(0.0, 0.0), omega=((1.0, 0.0), (0.0, 1.0)), alpha=(0.0, 0.0)):
    r"""f(x) = 2 phi_d(x - xi; Omega) Phi(alpha' w^{-1}(x - xi)), w = diag(Omega)^{1/2}.

    alpha = 0 gives the multivariate normal.

    Parameters
    ----------
    x : one point or list of points
    xi : location vector
    omega : d x d positive-definite scale matrix
    alpha : slant vector

    Returns
    -------
    RichResult
        Keys: pdf, logpdf.

    References
    ----------
    Azzalini, A. & Dalla Valle, A. (1996). Biometrika 83, 715-726.
    Matches ``sn::dmsn(x, xi, Omega, alpha)``.

    Examples
    --------
    >>> r = mvskewnorm([0.0, 0.0], alpha=[3.0, -1.0])
    >>> round(r["pdf"] * 2 * math.pi, 12)
    1.0
    """
    mu = [float(v) for v in xi]
    S = as_matrix(omega)
    a = [float(v) for v in alpha]
    d = len(mu)
    if len(S) != d or len(a) != d:
        raise ValueError("xi, omega and alpha dimensions differ")
    L = cholesky(S)
    ld = logdet_chol(L)
    w = [math.sqrt(S[i][i]) for i in range(d)]
    pts, single = points(x)
    lp = []
    for pt in pts:
        e = [p - m for p, m in zip(pt, mu)]
        z = forward(L, e)
        lin = sum(ai * ei / wi for ai, ei, wi in zip(a, e, w))
        cdf = pnorm(lin)
        lp.append(
            math.log(2)
            - 0.5 * d * math.log(2 * math.pi)
            - 0.5 * ld
            - 0.5 * sum(v * v for v in z)
            + (math.log(cdf) if cdf > 0 else -math.inf)
        )
    pdf = [math.exp(v) if v > -math.inf else 0.0 for v in lp]
    return RichResult(
        title="Multivariate skew-normal",
        summary_lines=[("dimension", d)],
        payload={"pdf": pdf[0] if single else pdf, "logpdf": lp[0] if single else lp},
    )


def cheatsheet():
    return "mvskewnorm: multivariate skew-normal density 2 phi_d Phi(alpha' w^-1 (x - xi))."
