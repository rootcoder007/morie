# morie.fn -- function file (rootcoder007/morie)
"""Null variance of Spearman's coefficient."""

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_spearman_rho_var"]


def gibbons_spearman_rho_var(n, r_s=None):
    r"""Null moments of Spearman's r_s and the standardised test.

    Under independence :math:`E(r_s) = 0` and, exactly,
    :math:`\mathrm{Var}(r_s) = 1/(n - 1)` (Gibbons Ch. 11.3) -- exact
    for every n, not merely asymptotic, which is what makes the
    normal approximation Z = r_s sqrt(n-1) so convenient.

    Parameters
    ----------
    n : int
        Sample size, at least 2.
    r_s : float, optional
        Observed coefficient; returns its z and p when given.

    Returns
    -------
    RichResult
        keys: ``var``, ``sd``, ``z``/``p_two_sided`` (if r_s given),
        ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.3.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    var = 1.0 / (n - 1)
    payload = {
        "var": float(var), "sd": float(np.sqrt(var)), "n": n,
        "method": "Var(r_s) = 1/(n-1), exact under the null (Gibbons Ch. 11.3)",
    }
    if r_s is not None:
        r_s = float(r_s)
        if not -1 <= r_s <= 1:
            raise ValueError(f"r_s must lie in [-1, 1], got {r_s}.")
        z = r_s / np.sqrt(var)
        payload["z"] = float(z)
        payload["p_two_sided"] = float(2 * stats.norm.sf(abs(z)))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb_spv: Var(r_s) = 1/(n-1), exact at every n"
