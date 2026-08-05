# morie.fn -- function file (rootcoder007/morie)
"""Upper tail dependence coefficient."""

from .chiDep import chi_dependence

from ._richresult import RichResult

__all__ = ["upper_tail_dependence"]


def upper_tail_dependence(y, copula, theta=0.95):
    """Upper tail dependence, estimated by Coles' chi(u) diagnostic.

    ``lambda_U = lim_{u->1-} (1 - 2u + C(u,u)) / (1 - u)`` is exactly the
    limit of Coles' ``chi(u) = 2 - log C(u,u) / log u``, so the estimator
    already in the tree is reused rather than written a second time; this
    module is a thin alias for ``chiDep.chi_dependence``.

    Formula: ``lambda_U = lim_{u->1-} (1 - 2u + C(u,u)) / (1 - u)``.

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
    Coles, S. (2001).  An Introduction to Statistical Modeling of Extreme
    Values.  Springer, section 8.4, pp. 163-165.
    """
    r = chi_dependence(y, copula, theta)
    return RichResult(payload={
        "estimate": float(r["estimate"]), "u": float(r["u"]), "n": int(r["n"]),
        "method": "upper tail dependence via empirical chi(u) [Joe 1997; Coles 2001]"})


# CANONICAL TEST
# >>> # comonotone data: the rank transforms coincide, so chi is exactly 1
# >>> assert abs(upper_tail_dependence(list(range(20)), list(range(20)), 0.5)["estimate"] - 1.0) < 1e-12


def cheatsheet():
    return "tldepu(y, copula, theta): upper tail dependence (alias of chiDep)."
