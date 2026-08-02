# morie.fn -- function file (rootcoder007/morie)
"""Chebyshev's inequality."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_chebyshev"]


def gibbons_chebyshev(k, x=None, mu=None, sigma=None):
    r"""Section 1.2.5: for ANY distribution with finite variance,

    .. math:: P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}.

    When data are supplied, the empirical exceedance frequency is
    computed alongside the bound so the slack is visible -- for
    near-normal data the true probability is far below 1/k^2, which
    is the price of universality.

    Parameters
    ----------
    k : float > 0
        Number of standard deviations.
    x : array-like, optional
        Data to check the bound against.
    mu, sigma : float, optional
        Location/scale; sample values if omitted.

    Returns
    -------
    RichResult
        keys: ``bound`` (min(1, 1/k^2)), ``empirical`` (if x given),
        ``k``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 1.2.5.
    """
    k = float(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    payload = {"bound": float(min(1.0, 1.0 / k**2)), "k": k,
               "method": "P(|X - mu| >= k sigma) <= 1/k^2 (Gibbons Ch. 1.2.5)"}
    if x is not None:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 2:
            raise ValueError("need at least 2 observations.")
        m = float(np.mean(x)) if mu is None else float(mu)
        s = float(np.std(x)) if sigma is None else float(sigma)
        if s <= 0:
            raise ValueError("sigma must be positive.")
        payload["empirical"] = float(np.mean(np.abs(x - m) >= k * s))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb_c1: 1/k^2 bound + empirical exceedance side by side"
