# morie.fn -- function file (rootcoder007/morie)
"""Monotone-treatment-selection bound."""

from . import _bndcore as B

from ._richresult import RichResult

__all__ = ["bound_skewed_outcome"]


def bound_skewed_outcome(y, D, skew=1.0):
    """ATE bound under monotone treatment selection.

    Monotone treatment selection says the treated would have done at
    least as well as the untreated under either treatment,
    ``E[y(t) | D = 1] >= E[y(t) | D = 0]``.  The consequence is one
    line of algebra: ``E[y(1)] = E[y | D = 1] P(D = 1) +
    E[y(1) | D = 0] P(D = 0) <= E[y | D = 1]``, and symmetrically
    ``E[y(0)] >= E[y | D = 0]``, so the naive observed difference, which
    without the assumption is not a bound on anything, becomes an exact
    upper bound on the ATE.  The other end is left at the worst case.

    Formula: with ``skew > 0``, ``[wc_lower, E(y | D = 1) - E(y | D = 0)]``;
    with ``skew < 0`` the inequality reverses and the observed difference
    is the lower bound.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.
    skew : float, optional
        Direction of the selection: positive for selection on gains
        (the default), negative for the reverse.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``naive``,
        ``wc_lower``, ``wc_upper``, ``p_treated``, ``n``.

    References
    ----------
    Manski, C. F. & Pepper, J. V. (2000).  Monotone instrumental
    variables, with an application to the returns to schooling.
    Econometrica 68(4), 997-1010.  doi:10.1111/1468-0262.00144.  The
    worst-case end is equation (2.11) of Molinari, F. (2021), Handbook of
    Econometrics 7A (arXiv:2004.11751 p. 17).  The two-line derivation
    above is written out because the paper itself was not accessible;
    what is cited to Manski and Pepper is the assumption, not a formula
    copied from them.
    """
    yv, dv = B.yd(y, D, "bound_skewed_outcome")
    p1, m1, p0, m0 = B.cellmeans(yv, dv)
    if p1 <= 0.0 or p0 <= 0.0:
        raise ValueError("bound_skewed_outcome: both treatment arms must be non-empty")
    y0, y1 = B.support(yv)
    wlo, whi = B.wc_ate(yv, dv, y0, y1)
    naive = m1 - m0
    s = float(skew)
    if s == 0.0:
        raise ValueError("bound_skewed_outcome: skew must be non-zero")
    if s > 0.0:
        lo, hi = wlo, naive
    else:
        lo, hi = naive, whi
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "naive": naive,
        "wc_lower": wlo, "wc_upper": whi, "p_treated": p1, "n": len(yv),
        "method": "Skewed-outcome bound"})


def cheatsheet():
    return "bndsdo: monotone-treatment-selection bound"
