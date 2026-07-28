# morie.fn -- function file (rootcoder007/morie)
"""Symmetry of linear rank statistics: folded scores."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_linrank_sym_special"]


def gibbons_linrank_sym_special(N):
    r"""Theorem 7.3.6: the folded scores

    .. math:: a_i = i \;(i \le N/2), \qquad
              a_i = N - i + 1 \;(i > N/2)

    make :math:`T_N` symmetric about its mean for any (m, n) --
    PROVIDED N IS EVEN. The book's proof builds the conjugate by
    swapping the two halves, :math:`Z_i' = Z_{i + N/2}` (PDF-verified,
    printed p. 282), which needs N/2 to be an integer; enumeration
    confirms the odd-N case is genuinely skewed (skewness 0.089 at
    N = 7, m = 2), so odd N raises rather than returning a wrong
    "symmetric".

    Parameters
    ----------
    N : int
        Number of ranks; must be even and at least 2.

    Returns
    -------
    RichResult
        keys: ``scores``, ``palindromic`` (a_i = a_{N-i+1}, the
        property the swap argument uses), ``symmetric``, ``N``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 7.3.6.
    """
    N = int(N)
    if N < 2:
        raise ValueError(f"N must be at least 2, got {N}.")
    if N % 2 != 0:
        raise ValueError(
            f"Theorem 7.3.6's half-swap conjugate needs even N, got {N}; "
            "the odd-N folded-score statistic is NOT symmetric "
            "(enumeration: skewness 0.089 at N = 7, m = 2)."
        )
    i = np.arange(1, N + 1)
    a = np.where(i <= N / 2.0, i, N - i + 1).astype(float)
    return RichResult(
        payload={
            "scores": a,
            "palindromic": bool(np.allclose(a, a[::-1])),
            "symmetric": True, "N": N,
            "method": "Folded scores, even N (Gibbons Theorem 7.3.6, p. 282)",
        }
    )


def cheatsheet():
    return "gb736: folded scores symmetric ONLY for even N (half-swap proof)"
