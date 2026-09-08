# morie.fn -- function file (rootcoder007/morie)
"""Glivenko-Cantelli supremum with the DKW bound."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["glivenko", "kosorok_glivenko_cantelli"]


def glivenko(x, F):
    """Uniform distance sup_t |F_n(t) - F(t)|, with the DKW tail bound.

    The supremum is over ALL real t but is attained only at the sample
    points, and at each one it must be evaluated on BOTH sides: F_n
    jumps there, so the largest gap may sit just below the jump rather
    than at it.  Checking only one side is the standard way to get a
    Kolmogorov-Smirnov statistic that is too small.

    Formula: D_n = sup_t |F_n(t) - F(t)|
                 = max_i max( i/n - F(x_(i)), F(x_(i)) - (i-1)/n );
             P(D_n > eps) <= 2 exp(-2 n eps^2)

    Parameters
    ----------
    x : array-like
        The sample.
    F : array-like
        The true cdf evaluated at the SORTED sample, same length as x.

    Returns
    -------
    RichResult
        ``statistic`` (D_n), ``d_plus``, ``d_minus``, ``argmax``
        (one-based index into the sorted sample), ``dkw_bound``,
        ``n``.

    References
    ----------
    Kosorok (2008), Introduction to Empirical Processes and
    Semiparametric Inference, Section 2.1, equation (2.3):
    sup_t |F_n(t) - F(t)| -> 0 almost surely (Glivenko 1933, Cantelli
    1933), with the general form (2.4) for a P-Glivenko-Cantelli class.
    Fetched as the full text of the book.  The constant 2 in the tail
    bound is Massart (1990), The tight constant in the
    Dvoretzky-Kiefer-Wolfowitz inequality, Annals of Probability 18(3),
    1269-1283; that sharp constant is NOT in Kosorok and is cited to its
    own source.
    """
    x = C.vec(x)
    F = C.vec(F)
    n = len(x)
    if n < 1:
        raise ValueError("the sample must be non-empty")
    if len(F) != n:
        raise ValueError("x and F must have the same length")
    idx = sorted(range(n), key=lambda i: x[i])
    Fs = [F[i] for i in idx]
    dp = 0.0
    dm = 0.0
    arg = 1
    best = -1.0
    for i in range(n):
        a = (i + 1) / n - Fs[i]
        b = Fs[i] - i / n
        dp = max(dp, a)
        dm = max(dm, b)
        if max(a, b) > best:
            best = max(a, b)
            arg = i + 1
    D = max(dp, dm)
    return RichResult(payload={
        "statistic": D, "d_plus": dp, "d_minus": dm, "argmax": float(arg),
        "dkw_bound": min(1.0, 2.0 * math.exp(-2.0 * n * D * D)),
        "n": float(n),
        "method": "Glivenko-Cantelli supremum with the DKW-Massart bound"})


kosorok_glivenko_cantelli = glivenko


def cheatsheet():
    return "ksr03: D_n = sup|F_n - F|, both sides of each jump; DKW 2exp(-2n eps^2)"
