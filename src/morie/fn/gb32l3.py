# morie.fn -- function file (rootcoder007/morie)
"""Shifted Vandermonde identity, Lemma 3.2.3."""

from math import comb

from ._richresult import RichResult

__all__ = ["gibbons_vandermonde_id2"]


def gibbons_vandermonde_id2(m, n):
    r"""Lemma 3.2.3:

    .. math:: \sum_{r=0}^{\min(m,\,n-1)}
              \binom{m}{r}\binom{n}{r+1} = \binom{m+n}{m+1}.

    The shifted companion of Lemma 3.2.2, used for the odd-total-runs
    terms where the two run counts differ by one.

    Parameters
    ----------
    m, n : int
        Non-negative integers, n >= 1.

    Returns
    -------
    RichResult
        keys: ``lhs``, ``rhs``, ``holds``, ``terms``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Lemma 3.2.3.
    """
    m, n = int(m), int(n)
    if m < 0 or n < 1:
        raise ValueError("need m >= 0 and n >= 1.")
    terms = [comb(m, r) * comb(n, r + 1) for r in range(min(m, n - 1) + 1)]
    lhs = sum(terms)
    rhs = comb(m + n, m + 1)
    return RichResult(
        payload={"lhs": lhs, "rhs": rhs, "holds": lhs == rhs,
                 "terms": terms, "m": m, "n": n,
                 "method": "sum C(m,r)C(n,r+1) = C(m+n,m+1) (Gibbons Lemma 3.2.3)"}
    )


def cheatsheet():
    return "gb32l3: shifted Vandermonde for the odd-runs terms"
