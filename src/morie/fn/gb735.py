# morie.fn -- function file (rootcoder007/morie)
"""Symmetry of linear rank statistics: equal sample sizes."""

from itertools import combinations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_linrank_sym_equal"]


def gibbons_linrank_sym_equal(a, m, n):
    r"""Theorem 7.3.5: when m = n, the two-sample linear rank
    statistic is symmetric about its mean for ANY score vector --
    swapping the sample labels is a measure-preserving bijection that
    reflects T about its mean, and equal sizes make the swap
    label-invariant.

    For small N the exact null distribution is enumerated as a
    demonstration; the returned skewness is 0 (to numerical
    precision) exactly when m = n or the scores are complementary.

    Parameters
    ----------
    a : array-like
        Scores a_1..a_N with N = m + n.
    m, n : int
        Sample sizes.

    Returns
    -------
    RichResult
        keys: ``symmetric``, ``mean``, ``skewness``, ``enumerated``
        (bool -- exact only for C(N, m) <= 200000), ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 7.3.5.
    """
    from math import comb

    a = np.asarray(a, dtype=float).ravel()
    m, n = int(m), int(n)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    N = m + n
    if a.size != N:
        raise ValueError(f"scores must have length m + n = {N}, got {a.size}.")
    if comb(N, m) > 200000:
        raise ValueError("N too large for exact enumeration; use the CLT route.")
    vals = np.array([a[list(idx)].sum() for idx in combinations(range(N), m)])
    mean = float(vals.mean())
    ctr = vals - mean
    sd = float(ctr.std())
    skew = float(np.mean(ctr**3) / sd**3) if sd > 0 else 0.0
    return RichResult(
        payload={
            "symmetric": bool(abs(skew) < 1e-10), "mean": mean,
            "skewness": skew, "enumerated": True, "m": m, "n": n,
            "method": "m = n forces T_N symmetric for any scores (Theorem 7.3.5)",
        }
    )


def cheatsheet():
    return "gb735: equal sizes -> symmetric T for ANY scores; verified by enumeration"
