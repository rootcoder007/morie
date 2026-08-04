# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic variance of the quantile estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["qasyvar", "fauzi_quantile_asymp_var"]


def qasyvar(p, n, density=None, qp=None):
    r"""Asymptotic variance of the quantile estimator.

    Eqs. (3.2)-(3.3):

    .. math:: V[\hat Q(p)] = \frac{p(1-p)}{n f^2(F^{-1}(p))} + O(n^{-2}),
              \qquad
              \mathrm{AMSE} = \frac{[Q'(p)]^2p(1-p)}{n},

    the second form using :math:`Q'(p) = 1/f(Q(p))`.

    Both spellings are returned, because the equality between them is the
    thing worth checking: a quantile's variance is the binomial variance
    :math:`p(1-p)` transported through the inverse-cdf map, and the
    density at the quantile is the Jacobian. Where the density is small
    the quantile is badly determined -- which is the entire reason tail
    quantiles are hard, stated as arithmetic.

    Remark 3.3 is blunt that the KERNEL quantile estimator has this same
    first-order variance. Chapter 3's improvement is second-order only,
    which is why its payoff has to be read off an Edgeworth expansion
    rather than an asymptotic variance.

    Supply ``qp`` (the quantile-function derivative) or ``density`` (the
    density at the quantile), not both.

    Parameters
    ----------
    p : float
        Probability in ``(0, 1)``.
    n : int
        Sample size.
    density : float, optional
        ``f(F^{-1}(p))``.
    qp : float, optional
        ``Q'(p)``; equals ``1 / density``.

    Returns
    -------
    RichResult
        Keys ``variance``, ``se``, ``amse``, ``sigma2``, ``qp``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (3.2)-(3.3).
    """
    p = float(p)
    n = int(n)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie strictly in (0, 1), got {p}.")
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if (density is None) == (qp is None):
        raise ValueError("supply exactly one of density or qp.")
    if qp is None:
        d = float(density)
        if d <= 0:
            raise ValueError("the density at the quantile must be positive.")
        qp = 1.0 / d
    qp = float(qp)
    sigma2 = qp * qp * p * (1.0 - p)
    var = sigma2 / n
    return RichResult(
        payload={
            "variance": float(var),
            "se": float(np.sqrt(var)),
            "amse": float(var),
            "sigma2": float(sigma2),
            "qp": qp,
            "n": int(n),
            "method": "asymptotic variance of the quantile estimator (Eqs. 3.2-3.3)",
        }
    )


fauzi_quantile_asymp_var = qasyvar


def cheatsheet():
    return "fzavar: p(1-p)/(n f^2(Q(p))) -- binomial variance through the inverse-cdf Jacobian (3.2)"


# CANONICAL TEST
# >>> r = qasyvar(p=0.5, n=100, density=0.4)
# >>> abs(r['variance'] - 0.25 / (100 * 0.16)) < 1e-15
# True
