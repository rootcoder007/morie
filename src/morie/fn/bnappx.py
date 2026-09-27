"""Normal approximation to the binomial distribution function with continuity correction."""

import math

from ._richresult import RichResult
from ._stats_core import binom, norm

__all__ = ["binomial_normal_approx"]


def binomial_normal_approx(x, n, p, correct=True):
    r"""Approximate :math:`P(X \le x)` for :math:`X \sim Bi(n, p)`.

    Hedderich, Sachs & Reynarowych (2023, eq 5.54):
    :math:`Z_1 = (x + 0.5 - np)/\sqrt{np(1-p)}`, :math:`P \approx \Phi(Z_1)`;
    ``correct=False`` drops the 0.5.

    Parameters
    ----------
    x : int
    n : int
    p : float
    correct : bool

    Returns
    -------
    RichResult
        ``z``, ``approx``, ``exact`` (``pbinom``).

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, eq (5.54).
    """
    n = int(n)
    p = float(p)
    if n < 1 or not 0 < p < 1:
        raise ValueError("need n >= 1 and 0 < p < 1")
    z = (x + (0.5 if correct else 0.0) - n * p) / math.sqrt(n * p * (1 - p))
    ap = float(norm.cdf(z))
    ex = float(binom.cdf(x, n, p))
    return RichResult(
        title="Normal approximation to the binomial",
        summary_lines=[("approx", ap), ("exact", ex)],
        payload={"z": z, "approx": ap, "exact": ex},
    )


def cheatsheet():
    return "bnappx: Phi((x + 0.5 - n p) / sqrt(n p (1 - p)))"
