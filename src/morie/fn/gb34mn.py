# morie.fn -- function file (rootcoder007/morie)
"""Moments of the runs up-and-down count."""

from ._richresult import RichResult

__all__ = ["gibbons_runs_ud_mean"]


def gibbons_runs_ud_mean(n):
    r"""Null moments of the number of runs up and down (Gibbons
    Ch. 3.4):

    .. math:: E(R_{ud}) = \frac{2n - 1}{3}, \qquad
              \mathrm{Var}(R_{ud}) = \frac{16n - 29}{90}.

    A run up or down is a maximal monotone stretch of the sequence of
    successive differences; there are n - 1 differences, so at most
    n - 1 runs. The mean follows from each interior difference sign
    changing with probability 2/3 under exchangeability.

    Parameters
    ----------
    n : int
        Sequence length, at least 3.

    Returns
    -------
    RichResult
        keys: ``mean``, ``var``, ``max_runs`` (n - 1), ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 3.4.
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"n must be at least 3, got {n}.")
    return RichResult(
        payload={"mean": (2.0 * n - 1) / 3.0, "var": (16.0 * n - 29) / 90.0,
                 "max_runs": n - 1, "n": n,
                 "method": "E = (2n-1)/3, Var = (16n-29)/90 (Gibbons Ch. 3.4)"}
    )


def cheatsheet():
    return "gb34mn: E(R_ud) = (2n-1)/3, Var = (16n-29)/90"
