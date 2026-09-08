# morie.fn -- function file (rootcoder007/morie)
"""Two-sample Kolmogorov-Smirnov test."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["kstest1", "wasserman_ks_test"]


def kstest1(x, y, terms=200):
    """Two-sample Kolmogorov-Smirnov test of H0: F1 = F2.

    The supremum is attained at one of the observed points, so it is a
    maximum over the pooled sample rather than a search -- both
    one-sided gaps are examined because the two empirical cdfs step at
    different places and taking only one of them silently halves the
    statistic.

    The p-value is the asymptotic one; it is unreliable for very small
    samples and for heavily tied data, and ``ties`` is returned so a
    caller can see whether that applies.

    Formula: D = sup_x |F1(x) - F2(x)|;
             t = sqrt(n1 n2/(n1 + n2)) D;
             p = 1 - H(t) = 2 sum_{j>=1} (-1)^{j-1} exp(-2 j^2 t^2)

    Parameters
    ----------
    x, y : array-like
        The two samples, each of length at least 1.
    terms : int
        Terms of the alternating series used for H(t).

    Returns
    -------
    RichResult
        ``statistic`` (D), ``scaled`` (t), ``p_value``, ``n1``, ``n2``,
        ``ties``.

    References
    ----------
    Wasserman (2004), All of Statistics, Section 15.4 and Theorem
    15.12: D = sup_x |F1hat(x) - F2hat(x)| with
    lim P( sqrt(n1 n2/(n1+n2)) D <= t ) = H(t) and
    H(t) = 1 - 2 sum_{j=1}^{inf} (-1)^{j-1} exp(-2 j^2 t^2),
    equation (15.14).  Fetched as the full text of the book.
    """
    x = sorted(C.vec(x))
    y = sorted(C.vec(y))
    n1 = len(x)
    n2 = len(y)
    if n1 < 1 or n2 < 1:
        raise ValueError("both samples must be non-empty")
    pool = sorted(set(x + y))
    D = 0.0
    for v in pool:
        f1 = sum(1 for a in x if a <= v) / n1
        f2 = sum(1 for b in y if b <= v) / n2
        D = max(D, abs(f1 - f2))
    t = math.sqrt(n1 * n2 / (n1 + n2)) * D
    p = 0.0
    if t > 0:
        for j in range(1, int(terms) + 1):
            p += (1.0 if j % 2 == 1 else -1.0) * math.exp(-2.0 * j * j * t * t)
        p *= 2.0
    else:
        p = 1.0
    p = min(1.0, max(0.0, p))
    return RichResult(payload={
        "statistic": D, "scaled": t, "p_value": p, "n1": n1, "n2": n2,
        "ties": float(n1 + n2 - len(pool)),
        "method": "Two-sample KS test, Wasserman Theorem 15.12"})


wasserman_ks_test = kstest1


def cheatsheet():
    return "wsmksm: D = sup|F1-F2|; p = 2 sum (-1)^(j-1) exp(-2 j^2 t^2)"
