# morie.fn -- function file (rootcoder007/morie)
"""Phi coefficient and Cramer's V."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["gibbons_phi_cramers_v"]


def gibbons_phi_cramers_v(table):
    r"""Section 14.2: chi-square-based association measures,

    .. math:: \phi = \sqrt{Q/n} \;(2\times 2), \qquad
              V = \sqrt{\frac{Q}{n \min(r - 1, c - 1)}}.

    V rescales phi so its maximum is 1 for ANY table shape; for a
    2x2 table the two coincide, which the tests assert.

    Parameters
    ----------
    table : array-like, shape (r, c)
        Observed counts.

    Returns
    -------
    RichResult
        keys: ``phi`` (None unless 2x2), ``cramers_v``, ``chi2``,
        ``n``, ``r``, ``c``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 14.2.

    Cramer, H. (1946). *Mathematical Methods of Statistics*.
    Princeton University Press. Sec. 21.9.
    """
    O = np.asarray(table, dtype=float)
    if O.ndim != 2 or O.shape[0] < 2 or O.shape[1] < 2:
        raise ValueError("table must be at least 2x2.")
    if np.any(O < 0):
        raise ValueError("counts must be non-negative.")
    r, c = O.shape
    ntot = O.sum()
    if ntot <= 0:
        raise ValueError("the table is empty.")
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / ntot
    if np.any(E == 0):
        raise ValueError("a margin is zero; the measures are degenerate.")
    Q = float(np.sum((O - E) ** 2 / E))
    V = float(np.sqrt(Q / (ntot * min(r - 1, c - 1))))
    phi = float(np.sqrt(Q / ntot)) if (r, c) == (2, 2) else None
    return RichResult(
        payload={
            "phi": phi, "cramers_v": V, "chi2": Q, "n": int(ntot),
            "r": int(r), "c": int(c),
            "method": "phi = sqrt(Q/n); V = sqrt(Q/(n min(r-1,c-1))) (Ch. 14.2)",
        }
    )


def cheatsheet():
    return "gb1421t: V rescales phi to max 1 for any shape; equal on 2x2"
