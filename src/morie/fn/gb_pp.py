# morie.fn -- function file (rootcoder007/morie)
"""P-P plot coordinates and summary."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_pp_plot"]


def gibbons_pp_plot(x, F0=None):
    r"""Section 4.8: the probability-probability plot pairs

    .. math:: \big(F_0(X_{(i)}),\; i/n\big),

    hypothesised probability against empirical probability. Under
    the null the points hug the diagonal; the maximum vertical
    departure IS the K-S statistic (up to the step convention), which
    the returned summary makes explicit. P-P plots resolve misfit in
    the CENTRE of the distribution where F changes fastest -- the
    complement of the Q-Q plot's tail sensitivity.

    Parameters
    ----------
    x : array-like
        Sample.
    F0 : callable, optional
        Hypothesised CDF; standard normal if omitted.

    Returns
    -------
    RichResult
        keys: ``theoretical`` (F0 at the order statistics),
        ``empirical`` (i/n), ``max_departure``, ``ks_equivalent``,
        ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 4.8.
    """
    from scipy import stats

    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    Fv = stats.norm.cdf(x) if F0 is None else np.asarray([F0(v) for v in x], dtype=float)
    emp = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    dep = float(np.max(np.maximum(np.abs(emp - Fv), np.abs(Fv - lo))))
    return RichResult(
        payload={
            "theoretical": Fv, "empirical": emp, "max_departure": dep,
            "ks_equivalent": dep, "n": int(n),
            "method": "P-P pairs (F0(X_(i)), i/n); max departure = D_n (Ch. 4.8)",
        }
    )


def cheatsheet():
    return "gb_pp: centre-sensitive; max vertical departure = K-S D_n"
