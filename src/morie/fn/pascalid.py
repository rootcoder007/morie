"""Pascal's rule: C(n, k) = C(n-1, k-1) + C(n-1, k).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (1.28), (1.60).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["pascalid"]


def pascalid(n, k):
    """Pascal's addition rule for binomial coefficients.

    Eq (1.28) states C(n,k) = C(n-1,k-1) + C(n-1,k).  Eq (1.60) proves it
    by writing both right-hand terms over factorials,
    (n-1)!/((k-1)!(n-k)!) + (n-1)!/(k!(n-k-1)!), whose common-denominator
    sum is n!/(k!(n-k)!).  Both branches are returned.

    Parameters
    ----------
    n, k : int
        Requires 1 <= k <= n - 1 so that both right-hand terms exist.

    Returns
    -------
    RichResult
        Keys: n, k, lhs, rhs, term_left, term_right, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.28), (1.60).
    """
    n = _morin._check_nonneg_int(n, "n")
    k = _morin._check_nonneg_int(k, "k")
    if k < 1 or k > n - 1:
        raise ValueError("Pascal's rule needs 1 <= k <= n - 1")
    left = _morin.binom(n - 1, k - 1)
    right = _morin.binom(n - 1, k)
    lhs = _morin.binom(n, k)
    if left + right != lhs:
        raise AssertionError("Pascal's rule failed")
    payload = {"n": float(n), "k": float(k), "lhs": float(lhs),
               "rhs": float(left + right), "term_left": float(left),
               "term_right": float(right), "forms_agree": 1.0}
    return RichResult(
        title="Pascal's rule C(n,k) = C(n-1,k-1) + C(n-1,k).",
        summary_lines=[("C(n,k)", lhs), ("sum", left + right)],
        payload=payload,
    )


def cheatsheet():
    return "pascalid: C(n,k) = C(n-1,k-1) + C(n-1,k), proved over factorials. Morin (2016) eqs (1.28), (1.60)."
