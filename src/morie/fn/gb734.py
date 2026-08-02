# morie.fn -- function file (rootcoder007/morie)
"""Symmetry of linear rank statistics: complementary scores."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_linrank_symmetry_cond"]


def gibbons_linrank_symmetry_cond(a, N=None):
    r"""Theorem 7.3.4: if the scores satisfy

    .. math:: a_i + a_{N-i+1} = c \quad\text{(constant in } i\text{)},

    the linear rank statistic :math:`T_N = \sum a_{R_i}` is symmetric
    about its null mean, so exact tables need only one tail. Checks
    the condition on the supplied scores and reports the constant.

    Parameters
    ----------
    a : array-like
        Score vector a_1..a_N.
    N : int, optional
        Length check.

    Returns
    -------
    RichResult
        keys: ``symmetric``, ``constant`` (c, or None), ``pair_sums``,
        ``N``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 7.3.4.
    """
    a = np.asarray(a, dtype=float).ravel()
    if N is not None and int(N) != a.size:
        raise ValueError(f"scores have length {a.size}, not N = {N}.")
    N = a.size
    if N < 2:
        raise ValueError("need at least 2 scores.")
    sums = a + a[::-1]
    sym = bool(np.allclose(sums, sums[0]))
    return RichResult(
        payload={
            "symmetric": sym, "constant": float(sums[0]) if sym else None,
            "pair_sums": sums, "N": int(N),
            "method": "a_i + a_{N-i+1} constant => T_N symmetric (Theorem 7.3.4)",
        }
    )


def cheatsheet():
    return "gb734: complementary scores sum constant -> one-tailed tables suffice"
