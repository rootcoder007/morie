# morie.fn -- function file (rootcoder007/morie)
"""Theorem 4.6: mean value property of boundary-free MRL estimators."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_theorem_4_6", "fauzi_thm4_6_mean_value"]


def fauzi_theorem_4_6(x, a1, mrl_at_a1, h=None):
    r"""Theorem 4.6 (Fauzi Eqs. 4.29-4.30): the mean value property,

    .. math:: \tilde m_{X,1}(a_1) + a_1 = \bar X + O_p(h^2),

    where :math:`a_1` is the lower boundary of the support.

    A sanity identity with real content: the mean residual life at
    the START of the support, plus that starting point, must be the
    overall mean, because everyone is still at risk there. The
    estimator satisfies it up to :math:`O(h^2)` -- it does not merely
    approximate the MRL pointwise, it reproduces a structural
    relationship the true MRL obeys exactly.

    That makes it a genuine diagnostic. A large discrepancy is
    evidence of a bandwidth too big or a transformation mismatched to
    the support, and the module returns the gap so it can be checked
    rather than assumed.

    Parameters
    ----------
    x : array-like
        The sample.
    a1 : float
        Lower boundary of the support.
    mrl_at_a1 : float
        The estimated MRL evaluated at ``a1``.
    h : float, optional
        Bandwidth, for reporting the expected O(h^2) tolerance.

    Returns
    -------
    RichResult
        keys: ``identity_lhs``, ``sample_mean``, ``gap``,
        ``expected_order``, ``within_expected``, ``a1``, ``n``,
        ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Theorem 4.6, Eqs. (4.29)-(4.30). From
    the PDF.
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    a = float(a1)
    if np.any(xv < a):
        raise ValueError(f"a1 = {a} is not a lower bound of the sample.")
    lhs = float(mrl_at_a1) + a
    xbar = float(xv.mean())
    gap = abs(lhs - xbar)
    tol = None if h is None else float(h) ** 2
    return RichResult(payload={
        "identity_lhs": lhs, "sample_mean": xbar, "gap": gap,
        "expected_order": "O(h^2)",
        "within_expected": None if tol is None else bool(gap <= 5 * tol),
        "a1": a, "n": int(n),
        "why_it_holds": "at the start of the support everyone is still at "
                        "risk, so MRL(a_1) + a_1 is the overall mean",
        "diagnostic_use": "a large gap indicates a bandwidth too big or a "
                          "transformation mismatched to the support",
        "method": "Theorem 4.6 (4.29): m_tilde(a_1) + a_1 = Xbar + O_p(h^2)"})


def cheatsheet():
    return "fzt46: MRL at the support's start plus that start IS the sample mean -- a real diagnostic"


#: Catalogue alias for :func:`fauzi_theorem_4_6`.
fauzi_thm4_6_mean_value = fauzi_theorem_4_6
