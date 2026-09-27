"""Fitting the gamma distribution by moments or maximum likelihood."""

import math

from ._richresult import RichResult
from ._sci_core import special

__all__ = ["gamma_fit"]


def gamma_fit(x, method="mle"):
    r"""Estimate the shape k and rate :math:`\lambda` of a gamma distribution.

    Method of moments (Hedderich, Sachs & Reynarowych 2023, eq 5.154):
    :math:`\hat k = n\bar x^2/\sum(x_i-\bar x)^2`,
    :math:`\hat\lambda = n\bar x/\sum(x_i-\bar x)^2`. Maximum likelihood: the
    shape solves :math:`\log k - \psi(k) = \log\bar x - \overline{\log x}`
    (solved here by bisection to machine precision) and
    :math:`\hat\lambda = \hat k/\bar x` -- ``MASS::fitdistr(x, "gamma")``.

    Parameters
    ----------
    x : sequence of float
        Positive observations.
    method : str
        ``"mle"`` or ``"moments"``.

    Returns
    -------
    RichResult
        ``shape``, ``rate``, ``scale``, ``method``.

    References
    ----------
    Choi, S. C. & Wette, R. (1969). Maximum likelihood estimation of the
    parameters of the gamma distribution and their bias. Technometrics 11,
    683-690. Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied
    Statistics: Methods Using R. Springer, eq (5.154).
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 2 or any(not v > 0 for v in xs):
        raise ValueError("need at least two positive observations")
    m = sum(xs) / n
    if method == "moments":
        ss = sum((v - m) ** 2 for v in xs)
        k, lam = n * m * m / ss, n * m / ss
    elif method == "mle":
        s = math.log(m) - sum(math.log(v) for v in xs) / n
        if not s > 0:
            raise ValueError("all observations equal; the shape is unbounded")
        lo, hi = 1e-10, 1.0
        f = lambda a: math.log(a) - float(special.digamma(a)) - s  # noqa: E731
        while f(hi) > 0:
            hi *= 2.0
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-15 * hi:
                break
        k = 0.5 * (lo + hi)
        lam = k / m
    else:
        raise ValueError("`method` must be 'mle' or 'moments'")
    return RichResult(
        title="Gamma distribution fit",
        summary_lines=[("shape", k), ("rate", lam)],
        payload={"shape": k, "rate": lam, "scale": 1.0 / lam, "method": method},
    )


def cheatsheet():
    return "gamfit: gamma shape and rate by moments (5.154) or maximum likelihood"
