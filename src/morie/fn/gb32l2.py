# morie.fn -- function file (rootcoder007/morie)
"""Vandermonde-type identity, Lemma 3.2.2."""

from math import comb

from ._richresult import RichResult

__all__ = ["gibbons_vandermonde_id1"]


def gibbons_vandermonde_id1(m, n):
    r"""Lemma 3.2.2:

    .. math:: \sum_{r=0}^{\min(m,n)} \binom{m}{r}\binom{n}{r}
              = \binom{m+n}{m}.

    Both sides are computed and returned, so the identity is
    *demonstrated* on the given (m, n) rather than merely asserted --
    this lemma is what collapses the double sum in the total-runs
    derivation.

    Parameters
    ----------
    m, n : int
        Non-negative integers.

    Returns
    -------
    RichResult
        keys: ``lhs``, ``rhs``, ``holds``, ``terms``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Lemma 3.2.2.
    """
    m, n = int(m), int(n)
    if m < 0 or n < 0:
        raise ValueError("m and n must be non-negative.")
    terms = [comb(m, r) * comb(n, r) for r in range(min(m, n) + 1)]
    lhs = sum(terms)
    rhs = comb(m + n, m)
    return RichResult(
        payload={"lhs": lhs, "rhs": rhs, "holds": lhs == rhs,
                 "terms": terms, "m": m, "n": n,
                 "method": "sum C(m,r)C(n,r) = C(m+n,m) (Gibbons Lemma 3.2.2)"}
    )


def cheatsheet():
    return "gb32l2: sum C(m,r)C(n,r) = C(m+n,m), computed both sides"
