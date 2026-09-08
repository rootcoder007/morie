# morie.fn -- function file (rootcoder007/morie)
"""Compound-outcome worst-case bound."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_compound_outcome"]


def bound_compound_outcome(y_components, D, X):
    """Worst-case ATE bounds for a weighted composite of several outcomes.

    A composite endpoint is a fixed linear functional of its components, so
    the identification problem is the ordinary one for the scalar
    ``sum_k w_k y_k``.  The support of the composite is read off the
    realised composite values rather than combined component-wise, which
    keeps the interval sharp for the data at hand.

    Formula: ``c_i = sum_k w_k y_ik``, then the worst-case bound of
    Molinari (2021) eq. (2.11) applied to ``c``.

    Parameters
    ----------
    y_components : array-like, shape (n, k)
        Outcome components, one row per unit.
    D : array-like
        Binary treatment indicator, coded 0/1.
    X : array-like, length k
        Component weights.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``k``, ``n``.

    References
    ----------
    Manski, C. F. (2003).  Partial Identification of Probability
    Distributions.  Springer, New York.  Worst-case form as equation (2.11)
    of Molinari, F. (2021), Handbook of Econometrics 7A
    (arXiv:2004.11751 p. 17).
    """
    M = C.mat(y_components)
    n = len(M)
    if n == 0:
        raise ValueError("bound_compound_outcome: y_components is empty")
    k = len(M[0])
    for r in M:
        if len(r) != k:
            raise ValueError("bound_compound_outcome: ragged component matrix")
    w = C.vec(X)
    if len(w) != k:
        raise ValueError("bound_compound_outcome: X must give one weight per component")
    comp = []
    for i in range(n):
        s = 0.0
        for j in range(k):
            s += w[j] * M[i][j]
        comp.append(s)
    cv, dv = B.yd(comp, D, "bound_compound_outcome")
    y0, y1 = B.support(cv)
    lo, hi = B.wc_ate(cv, dv, y0, y1)
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "k": k, "n": n,
        "method": "Compound-outcome bound"})


def cheatsheet():
    return "bnscbo: Compound-outcome bound"
