# morie.fn -- function file (rootcoder007/morie)
"""Chi-square test with Yates's continuity correction."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["gibbons_chi2_yates"]


def gibbons_chi2_yates(table, cdf=None):
    r"""Yates-corrected chi-square for a 2x2 table:

    .. math:: Q_c = \sum_{ij} \frac{(|O_{ij} - E_{ij}| - 0.5)^2}
              {E_{ij}},

    the cellwise version of the half-unit continuity correction.
    Restricted to 2x2 -- Yates's argument is about a single discrete
    hypergeometric margin, and applying it to larger tables
    over-corrects, so bigger tables raise rather than silently
    getting the wrong adjustment.

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Observed counts.
    cdf : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``chi2_corrected``, ``chi2_uncorrected``, ``df`` (1),
        ``p_value``, ``expected``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 14.2.

    Yates, F. (1934). Contingency tables involving small numbers and
    the chi-square test. *Supplement to the Journal of the Royal
    Statistical Society*, 1(2), 217-235.
    """
    O = np.asarray(table, dtype=float)
    if O.shape != (2, 2):
        raise ValueError(
            f"Yates's correction is a 2x2 argument; got shape {O.shape}."
        )
    if np.any(O < 0):
        raise ValueError("counts must be non-negative.")
    ntot = O.sum()
    if ntot <= 0:
        raise ValueError("the table is empty.")
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / ntot
    if np.any(E == 0):
        raise ValueError("a margin is zero; the test is degenerate.")
    qc = float(np.sum((np.maximum(np.abs(O - E) - 0.5, 0.0)) ** 2 / E))
    q0 = float(np.sum((O - E) ** 2 / E))
    return RichResult(
        payload={
            "chi2_corrected": qc, "chi2_uncorrected": q0, "df": 1,
            "p_value": float(stats.chi2.sf(qc, 1)), "expected": E,
            "method": "Yates Q_c = sum (|O - E| - .5)^2/E, 2x2 only (Ch. 14.2)",
        }
    )


def cheatsheet():
    return "gb_c2: Yates on 2x2 only; larger tables over-correct"
