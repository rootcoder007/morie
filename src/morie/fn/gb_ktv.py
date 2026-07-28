# morie.fn -- function file (rootcoder007/morie)
"""Null variance of Kendall's tau."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_kendall_tau_var"]


def gibbons_kendall_tau_var(n):
    r"""Exact null variance of Kendall's T.

    .. math:: \mathrm{Var}(T) = \frac{2(2n + 5)}{9 n (n - 1)}

    (Gibbons Ch. 11.2). E(T) = 0 under independence. This is the
    variance of the *coefficient* T in [-1, 1], not of the raw score
    S = P - Q; the score variance n(n-1)(2n+5)/18 is also returned so
    the two never get conflated.

    Parameters
    ----------
    n : int
        Sample size, at least 2.

    Returns
    -------
    RichResult
        keys: ``var_tau``, ``sd_tau``, ``var_score``
        (n(n-1)(2n+5)/18), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.2.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    var_tau = 2.0 * (2 * n + 5) / (9.0 * n * (n - 1))
    return RichResult(
        payload={
            "var_tau": float(var_tau), "sd_tau": float(np.sqrt(var_tau)),
            "var_score": float(n * (n - 1) * (2 * n + 5) / 18.0),
            "n": n, "method": "Var(T) = 2(2n+5)/(9n(n-1)) (Gibbons Ch. 11.2)",
        }
    )


def cheatsheet():
    return "gb_ktv: Var(T) = 2(2n+5)/(9n(n-1)); score variance kept separate"
