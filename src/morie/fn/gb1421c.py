# morie.fn -- function file (rootcoder007/morie)
"""Pearson's contingency coefficient."""

from . import _array_core as np

from ._richresult import RichResult
from .gb1421t import gibbons_phi_cramers_v

__all__ = ["gibbons_contingency_coeff"]


def gibbons_contingency_coeff(table):
    r"""Section 14.2.1: Pearson's contingency coefficient,

    .. math:: C = \sqrt{\frac{\chi^2}{\chi^2 + n}} \in [0, 1),

    with the awkward property that its MAXIMUM depends on the table
    shape: for a k x k table, :math:`C_{\max} = \sqrt{(k-1)/k}` < 1.
    The shape-adjusted ratio C/C_max is returned alongside, since
    comparing raw C across differently shaped tables is a category
    error the raw number invites.

    Parameters
    ----------
    table : array-like, shape (r, c)
        Observed counts.

    Returns
    -------
    RichResult
        keys: ``C``, ``C_max`` (sqrt((k-1)/k), k = min(r, c)),
        ``C_adjusted`` (C/C_max), ``chi2``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 14.2.1.
    """
    out = gibbons_phi_cramers_v(table)
    Q, ntot = out["chi2"], out["n"]
    C = float(np.sqrt(Q / (Q + ntot)))
    k = min(out["r"], out["c"])
    Cmax = float(np.sqrt((k - 1) / k))
    return RichResult(
        payload={
            "C": C, "C_max": Cmax, "C_adjusted": C / Cmax, "chi2": Q,
            "n": ntot,
            "method": "C = sqrt(chi2/(chi2 + n)); max depends on shape (Ch. 14.2.1)",
        }
    )


def cheatsheet():
    return "gb1421c: C < sqrt((k-1)/k) always; adjusted ratio returned"
