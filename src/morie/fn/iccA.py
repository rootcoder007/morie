# morie.fn -- function file (rootcoder007/morie)
"""Intraclass correlation, absolute agreement."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["icc_absolute_agreement"]


def icc_absolute_agreement(y, subject, rater):
    """Agreement that counts a systematic rater offset as disagreement.

    The extra ``k (MS_C - MS_E) / n`` term in the denominator is the
    whole difference from the consistency form: it charges for
    between-rater bias.  Two raters who agree perfectly on the ordering
    but differ by a constant score high on consistency and low here, and
    which of those is the honest number depends entirely on whether the
    scale means anything.

    Formula: ``ICC(A,1) = (MS_R - MS_E) /
    [MS_R + (k - 1) MS_E + k (MS_C - MS_E) / n]``.

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
    McGraw, K. O. & Wong, S. P. (1996).  Forming inferences about some
    intraclass correlation coefficients.  Psychological Methods
    1:30-46, table 4.  The underlying model is Shrout, P. E. & Fleiss,
    J. L. (1979), Psychological Bulletin 86:420-428.
    """
    ms = S.icc_ms(y, subject, rater)
    k, n = ms["k"], ms["n"]
    den = ms["ms_r"] + (k - 1.0) * ms["ms_e"] + k * (ms["ms_c"] - ms["ms_e"]) / n
    return RichResult(payload={
        "estimate": (ms["ms_r"] - ms["ms_e"]) / den if den != 0.0 else float("nan"),
        "ms_r": ms["ms_r"], "ms_c": ms["ms_c"], "ms_e": ms["ms_e"],
        "k": k, "n_subjects": n,
        "method": "Intraclass correlation ICC(A,1)"})


def cheatsheet():
    return "iccA: Intraclass correlation, absolute agreement."
