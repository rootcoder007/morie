# morie.fn -- function file (rootcoder007/morie)
"""Distribution of the sample range."""

from . import _array_core as np
from ._sci_core import integrate

from ._richresult import RichResult

__all__ = ["gibbons_range_dist"]


def gibbons_range_dist(w, n, f=None, F=None):
    r"""Section 2.7.2: the CDF of the range W = X_(n) - X_(1) is

    .. math:: F_W(w) = n \int_{-\infty}^{\infty}
              [F(x + w) - F(x)]^{n-1} f(x)\, dx, \qquad w > 0,

    integrating over the position of the minimum. Evaluated by
    adaptive quadrature for any supplied density/CDF pair (standard
    normal by default).

    Parameters
    ----------
    w : float
        Range value, w > 0.
    n : int
        Sample size, at least 2.
    f, F : callable, optional
        Parent density and CDF; standard normal if omitted.

    Returns
    -------
    RichResult
        keys: ``cdf``, ``w``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 2.7.2.
    """
    from . import _stats_core as stats

    w = float(w)
    if w <= 0:
        raise ValueError(f"w must be positive, got {w}.")
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    if f is None:
        f = stats.norm.pdf
    if F is None:
        F = stats.norm.cdf

    val, _ = integrate.quad(
        lambda t: (F(t + w) - F(t)) ** (n - 1) * f(t), -np.inf, np.inf, limit=200
    )
    return RichResult(
        payload={
            "cdf": float(min(max(n * val, 0.0), 1.0)), "w": w, "n": n,
            "method": "F_W(w) = n int [F(x+w)-F(x)]^{n-1} f(x) dx (Ch. 2.7.2)",
        }
    )


def cheatsheet():
    return "gb_rng: range CDF by quadrature over the minimum's position"
