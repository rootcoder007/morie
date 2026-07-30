# morie.fn -- function file (rootcoder007/morie)
"""Theorem 4.5: strong uniform consistency of boundary-free MRL estimators."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_theorem_4_5", "fauzi_thm4_5_mrl_consistency"]


def fauzi_theorem_4_5(mrl_hat, mrl_true, t_grid, interval=None):
    r"""Theorem 4.5 (Fauzi): strong uniform consistency,

    .. math:: \sup_{t \in B}\big|\tilde m_{X,i}(t) - m_X(t)\big|
              \to_{a.s.} 0

    on a bounded interval :math:`B`.

    UNIFORM and ALMOST SURE, which is stronger than the pointwise
    convergence in distribution of Theorem 4.4 and is what licenses
    using the estimated curve as a whole -- for a plot, a maximum, or
    a crossing point -- rather than at one pre-chosen t.

    Lemma 4.3's argument is worth noting: it gets uniformity from
    MONOTONICITY plus pointwise convergence on a finite grid, the
    same device that proves the Glivenko-Cantelli theorem. Bounded
    ``B`` is required; on an unbounded interval the supremum need not
    converge.

    Parameters
    ----------
    mrl_hat, mrl_true : array-like
        Estimated and true MRL on ``t_grid``.
    t_grid : array-like
        Evaluation points.
    interval : tuple, optional
        The bounded ``B`` to take the supremum over.

    Returns
    -------
    RichResult
        keys: ``sup_error``, ``argmax_t``, ``interval``,
        ``mode`` ("uniform, almost sure"), ``requires_bounded_B``
        (True), ``stronger_than_pointwise`` (True), ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Theorem 4.5 and Lemma 4.3. From the PDF.
    """
    mh = np.atleast_1d(np.asarray(mrl_hat, dtype=float)).ravel()
    mt = np.atleast_1d(np.asarray(mrl_true, dtype=float)).ravel()
    tg = np.atleast_1d(np.asarray(t_grid, dtype=float)).ravel()
    if not (mh.size == mt.size == tg.size):
        raise ValueError("all three arguments must have the same length.")
    if interval is None:
        sel = np.ones(tg.size, dtype=bool)
        iv = (float(tg.min()), float(tg.max()))
    else:
        lo, hi = float(interval[0]), float(interval[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("the interval must be bounded with lo < hi; "
                             "uniform consistency is stated on a BOUNDED B.")
        sel = (tg >= lo) & (tg <= hi)
        iv = (lo, hi)
    if not np.any(sel):
        raise ValueError("no grid points fall inside the interval.")
    err = np.abs(mh[sel] - mt[sel])
    k = int(np.nanargmax(err))
    return RichResult(payload={
        "sup_error": float(np.nanmax(err)), "argmax_t": float(tg[sel][k]),
        "interval": iv, "mode": "uniform, almost sure",
        "requires_bounded_B": True, "stronger_than_pointwise": True,
        "proof_device": "monotonicity plus pointwise convergence on a finite "
                        "grid, as in Glivenko-Cantelli",
        "licenses": "using the whole estimated curve -- a maximum, a crossing "
                    "point -- not just one pre-chosen t",
        "method": "Theorem 4.5: strong uniform consistency on a bounded interval"})


def cheatsheet():
    return "fzt45: uniform + a.s. is what lets you use the WHOLE curve, not one point"


#: Catalogue alias for :func:`fauzi_theorem_4_5`.
fauzi_thm4_5_mrl_consistency = fauzi_theorem_4_5
