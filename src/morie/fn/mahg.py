# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hedges' g, the bias-corrected standardised mean difference.

Source consulted: Hedges, L.V. (1981). Distribution theory for Glass's
estimator of effect size and related estimators.  *Journal of Educational
Statistics* 6(2), 107-128.  Glass's estimator g = (m1 - m2)/s_pooled is
biased upward in small samples; Hedges' equation (6e) gives the exact
correction factor

    c(m) = Gamma(m/2) / ( sqrt(m/2) Gamma((m-1)/2) ),   m = n1 + n2 - 2

and the unbiased estimator is g_U = c(m) g, which the paper further shows is
the unique minimum-variance unbiased estimator of the population effect.  The
paper also gives the algebraic approximation c(m) ~= 1 - 3/(4m - 1), stated
there to have a maximum error of 0.007 at m = 2 and under 1.5e-5 for m > 50;
it is reported here as ``J_approx`` but not used.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_hedges_g"]


def _lgamma(x):
    """Log-gamma by the Lanczos approximation (g = 7, n = 9)."""
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if x < 0.5:
        return float(np.log(np.pi / abs(np.sin(np.pi * x)))) - _lgamma(1.0 - x)
    x = x - 1.0
    a = c[0]
    t = x + 7.5
    for i in range(1, 9):
        a += c[i] / (x + i)
    return 0.5 * float(np.log(2.0 * np.pi)) + (x + 0.5) * float(np.log(t)) - t + float(np.log(a))


def ma_hedges_g(m1, m2, s1, s2, n1, n2):
    """Hedges' g with the exact small-sample correction.

    Parameters
    ----------
    m1, m2 : float
        Group means.
    s1, s2 : float
        Group standard deviations (divisor n - 1).
    n1, n2 : int
        Group sizes.

    Returns
    -------
    RichResult
        estimate (g), d (uncorrected), J, J_approx, se, variance, df,
        s_pooled, n, method.

    References
    ----------
    Hedges (1981), J. Educational Statistics 6(2), 107-128, eq. (6e).
    """
    n1f = float(n1)
    n2f = float(n2)
    df = n1f + n2f - 2.0
    sp = float(np.sqrt(((n1f - 1.0) * float(s1) ** 2 + (n2f - 1.0) * float(s2) ** 2) / df))
    d = (float(m1) - float(m2)) / sp
    jexact = float(np.exp(_lgamma(df / 2.0) - 0.5 * float(np.log(df / 2.0)) - _lgamma((df - 1.0) / 2.0)))
    japprox = 1.0 - 3.0 / (4.0 * df - 1.0)
    g = jexact * d
    var = (n1f + n2f) / (n1f * n2f) + g * g / (2.0 * (n1f + n2f))
    return RichResult(
        payload={
            "estimate": float(g),
            "d": float(d),
            "J": float(jexact),
            "J_approx": float(japprox),
            "se": float(np.sqrt(var)),
            "variance": float(var),
            "df": float(df),
            "s_pooled": float(sp),
            "n": int(n1 + n2),
            "method": "Hedges g, bias-corrected standardised mean difference (Hedges 1981)",
        }
    )


# CANONICAL TEST
# >>> # equal means give zero effect whatever the correction
# >>> r = ma_hedges_g(5.0, 5.0, 1.0, 1.0, 10, 10)
# >>> assert abs(r["estimate"]) < 1e-15
# >>> # the exact and approximate corrections agree closely for df = 18
# >>> assert abs(r["J"] - r["J_approx"]) < 1e-4


def cheatsheet():
    return "mahg(m1, m2, s1, s2, n1, n2): Hedges g with exact correction."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mahedgesg = ma_hedges_g
