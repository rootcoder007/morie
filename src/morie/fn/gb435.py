# morie.fn -- function file (rootcoder007/morie)
"""One-sided K-S asymptotic distribution."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_ks_onesided_asymp"]


def gibbons_ks_onesided_asymp(d, n=None):
    r"""Theorem 4.3.5: the one-sided statistic has the exponential
    limit

    .. math:: \lim_{n\to\infty} P(D_n^+ \le d/\sqrt n)
              = 1 - e^{-2 d^2},

    i.e. :math:`2\sqrt n D_n^+` is asymptotically Rayleigh -- a far
    simpler limit than the two-sided Kolmogorov series because only
    one boundary can be crossed.

    Parameters
    ----------
    d : float > 0
        Argument (or raw statistic when n is given).
    n : int, optional
        Sample size for the finite-n evaluation at d sqrt(n).

    Returns
    -------
    RichResult
        keys: ``cdf``, ``p_value`` (exp(-2 d^2)), ``finite_n_cdf``
        (if n given), ``d``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 4.3.5.

    Smirnov, N. V. (1939). Sur les ecarts de la courbe de
    distribution empirique. *Matematicheskii Sbornik*, 6(48), 3-26.
    """
    d = float(d)
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")
    payload = {
        "cdf": float(1.0 - np.exp(-2.0 * d**2)),
        "p_value": float(np.exp(-2.0 * d**2)), "d": d,
        "method": "P(D+ <= d/sqrt(n)) -> 1 - exp(-2 d^2) (Theorem 4.3.5)",
    }
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        payload["finite_n_cdf"] = float(1.0 - np.exp(-2.0 * n * d**2))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb435: one-sided limit 1 - exp(-2d^2); Rayleigh in disguise"
