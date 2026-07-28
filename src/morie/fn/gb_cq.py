# morie.fn -- function file (rootcoder007/morie)
"""Cramer's V (front-end)."""

import numpy as np

from ._richresult import RichResult
from .gb1421t import gibbons_phi_cramers_v

__all__ = ["gibbons_cramers_contingency"]


def gibbons_cramers_contingency(table):
    r"""Cramer's V, delegating to :mod:`morie.fn.gb1421t`:

    .. math:: V = \sqrt{\frac{\chi^2}{n \min(r - 1, c - 1)}}.

    V = 0 at exact independence; V = 1 when every row (or column)
    concentrates in a single cell -- attainable for every table
    shape, which the raw phi cannot claim.

    Parameters
    ----------
    table : array-like, shape (r, c)
        Observed counts.

    Returns
    -------
    RichResult
        keys: ``cramers_v``, ``chi2``, ``n``, ``r``, ``c``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 14.2.
    """
    out = gibbons_phi_cramers_v(table)
    return RichResult(
        payload={
            "cramers_v": out["cramers_v"], "chi2": out["chi2"], "n": out["n"],
            "r": out["r"], "c": out["c"],
            "method": "Cramer's V = sqrt(chi2/(n min(r-1, c-1))) (Ch. 14.2)",
        }
    )


def cheatsheet():
    return "gb_cq: V front-end; delegates to gb1421t"
