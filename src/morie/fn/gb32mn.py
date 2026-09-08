# morie.fn -- function file (rootcoder007/morie)
"""Mean of the total number of runs."""

from ._richresult import RichResult

__all__ = ["gibbons_runs_mean"]


def gibbons_runs_mean(n1, n2):
    r"""Null mean of the total runs count (Gibbons eq. 3.2.6):

    .. math:: E(R) = 1 + \frac{2 n_1 n_2}{n_1 + n_2}.

    Derived from R = 1 + sum of change indicators I_k, each with
    P(I_k = 1) = 2 n_1 n_2 / [n(n-1)] -- the indicator route the
    chapter uses because the direct factorial-moment algebra is, in
    the book's own words, tedious.

    Parameters
    ----------
    n1, n2 : int
        Counts of each type, both >= 1.

    Returns
    -------
    RichResult
        keys: ``mean``, ``n1``, ``n2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Eq. (3.2.6).
    """
    n1, n2 = int(n1), int(n2)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    return RichResult(
        payload={"mean": 1.0 + 2.0 * n1 * n2 / (n1 + n2), "n1": n1, "n2": n2,
                 "method": "E(R) = 1 + 2 n1 n2/(n1+n2) (Gibbons eq. 3.2.6)"}
    )


def cheatsheet():
    return "gb32mn: E(R) = 1 + 2 n1 n2/n via change indicators"
