# morie.fn -- function file (rootcoder007/morie)
"""Intraclass correlation, consistency."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["icc_consistency"]


def icc_consistency(y, subject, rater):
    """Agreement up to a constant offset per rater.

    The consistency form deliberately does not penalise a rater who is
    systematically high, because the column mean square is left out of
    the denominator.  That is the right choice when only the ordering
    matters -- ranking candidates, say -- and the wrong one when the
    absolute number is the point.

    Formula: ``ICC(C,1) = (MS_R - MS_E) / (MS_R + (k - 1) MS_E)`` from
    the two-way mixed model.

    Parameters
    ----------
    y : array-like, shape (n,)
        Ratings.
    subject : array-like, shape (n,)
        Subject label.
    rater : array-like, shape (n,)
        Rater label.

    Returns
    -------
    RichResult
        ``estimate``, ``ms_r``, ``ms_c``, ``ms_e``, ``k``,
        ``n_subjects``.

    References
    ----------
    Shrout, P. E. & Fleiss, J. L. (1979).  Intraclass correlations: uses
    in assessing rater reliability.  Psychological Bulletin 86:420-428.
    The (C, 1) and (A, 1) naming is McGraw, K. O. & Wong, S. P. (1996),
    Forming inferences about some intraclass correlation coefficients,
    Psychological Methods 1:30-46.
    """
    ms = S.icc_ms(y, subject, rater)
    den = ms["ms_r"] + (ms["k"] - 1.0) * ms["ms_e"]
    return RichResult(payload={
        "estimate": (ms["ms_r"] - ms["ms_e"]) / den if den != 0.0 else float("nan"),
        "ms_r": ms["ms_r"], "ms_c": ms["ms_c"], "ms_e": ms["ms_e"],
        "k": ms["k"], "n_subjects": ms["n"],
        "method": "Intraclass correlation ICC(C,1)"})


iccconsistency = icc_consistency


def cheatsheet():
    return "iccC: Intraclass correlation, consistency."
