# morie.fn -- function file (rootcoder007/morie)
"""Variance of the total number of runs."""

from ._richresult import RichResult

__all__ = ["gibbons_runs_var"]


def gibbons_runs_var(n1, n2):
    r"""Null variance of the total runs count (Gibbons eq. 3.2.8):

    .. math:: \mathrm{Var}(R) = \frac{2 n_1 n_2 (2 n_1 n_2 - n_1 -
              n_2)}{(n_1 + n_2)^2 (n_1 + n_2 - 1)}.

    Together with eq. (3.2.6) this drives the large-sample normal
    runs test; the variance vanishes when either type has a single
    element and the arrangement is nearly forced.

    Parameters
    ----------
    n1, n2 : int
        Counts of each type, both >= 1, n1 + n2 >= 2.

    Returns
    -------
    RichResult
        keys: ``var``, ``sd``, ``n1``, ``n2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Eq. (3.2.8).
    """
    import numpy as np

    n1, n2 = int(n1), int(n2)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    n = n1 + n2
    if n < 3:
        raise ValueError("need n1 + n2 >= 3 for a non-degenerate variance.")
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / (n**2 * (n - 1))
    return RichResult(
        payload={"var": float(var), "sd": float(np.sqrt(max(var, 0.0))),
                 "n1": n1, "n2": n2,
                 "method": "Var(R) = 2n1n2(2n1n2-n1-n2)/[n^2(n-1)] (eq. 3.2.8)"}
    )


def cheatsheet():
    return "gb32vr: eq. 3.2.8; vanishes when one type is a single element"
