# morie.fn -- function file (rootcoder007/morie)
"""Lower tail dependence coefficient."""

from . import _tail1core as C
from .chiDep import chi_dependence

from ._richresult import RichResult

__all__ = ["lower_tail_dependence"]


def lower_tail_dependence(y, copula, theta=0.95):
    """Lower tail dependence, via the reflected chi(u) diagnostic.

    ``lambda_L = lim_{u->0+} C(u,u) / u``.  The lower tail of ``(X, Y)``
    is the upper tail of ``(-X, -Y)``, so this is Coles' ``chi(u)``
    applied to the reflected sample; no second estimator is written.  For
    a radially symmetric dependence structure the lower and upper
    coefficients coincide, which is the anchor used in the tests.

    Formula: ``lambda_L = lim_{u->0+} C(u,u) / u``.

    Parameters
    ----------
    y : array-like
        First margin.
    copula : array-like
        Second margin, equal length.
    theta : float, default 0.95
        Threshold ``u`` in (0, 1).

    Returns
    -------
    RichResult
        ``estimate``, ``u``, ``n``, ``method``.

    References
    ----------
    Joe, H. (1997).  Multivariate Models and Multivariate Dependence
    Concepts.  Chapman & Hall, section 2.1.10.
    """
    a = [-v for v in C.vec(y)]
    b = [-v for v in C.vec(copula)]
    r = chi_dependence(a, b, theta)
    return RichResult(payload={
        "estimate": float(r["estimate"]), "u": float(r["u"]), "n": int(r["n"]),
        "method": "lower tail dependence via reflected empirical chi(u) [Joe 1997]"})


# CANONICAL TEST
# >>> # comonotone data is radially symmetric here: both tails give exactly 1
# >>> assert abs(lower_tail_dependence(list(range(20)), list(range(20)), 0.5)["estimate"] - 1.0) < 1e-12


def cheatsheet():
    return "tldepl(y, copula, theta): lower tail dependence (reflected chiDep)."
