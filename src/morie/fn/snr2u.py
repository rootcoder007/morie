# morie.fn -- function file (rootcoder007/morie)
"""Snijders-Bosker level-2 explained variance for a two-level model."""

from ._richresult import RichResult

__all__ = ["snijders_bosker_r2_level2"]


def snijders_bosker_r2_level2(sigma2_e1, sigma2_u1, sigma2_e0, sigma2_u0, n=1.0):
    """Proportion of level-2 variance explained, relative to an empty model.

    At level two the quantity being predicted is a *group mean*, whose
    residual variance is ``sigma2_u + sigma2_e / n`` rather than
    ``sigma2_u + sigma2_e``: averaging over ``n`` members shrinks the
    level-1 noise by ``n`` but leaves the group-level term untouched.
    Everything else is as at level one.  With ``n = 1`` the two measures
    coincide exactly, and as ``n`` grows the measure tends to
    ``1 - sigma2_u1 / sigma2_u0``; both are used as anchors.

    Formula:
    ``R2_2 = 1 - (sigma2_e1/n + sigma2_u1) / (sigma2_e0/n + sigma2_u0)``.

    Parameters
    ----------
    sigma2_e1, sigma2_u1 : float
        Variance components of the fitted model.  Non-negative.
    sigma2_e0, sigma2_u0 : float
        Variance components of the empty baseline model.
    n : float, default 1
        Group size.  For unbalanced data Snijders and Bosker use a
        representative value; the harmonic mean of the group sizes is the
        usual choice.  Must be positive.

    Returns
    -------
    RichResult
        ``estimate`` (R2_2), ``total1``, ``total0``, ``n``,
        ``limit_large_n`` (the ``n -> infinity`` value),
        ``reduction``, ``method``.

    References
    ----------
    Snijders, T. A. B. & Bosker, R. J. (1994).  Modeled variance in
    two-level models.  Sociological Methods & Research 22(3):342-363.
    <https://doi.org/10.1177/0049124194022003004>
    Snijders, T. A. B. & Bosker, R. J. (2012).  Multilevel Analysis, 2nd
    edition, Sage, section 7.2.
    """
    e1, u1, e0, u0 = float(sigma2_e1), float(sigma2_u1), float(sigma2_e0), float(sigma2_u0)
    nn = float(n)
    for v in (e1, u1, e0, u0):
        if v < 0.0:
            raise ValueError("snijders_bosker_r2_level2: variance components must be non-negative")
    if nn <= 0.0:
        raise ValueError("snijders_bosker_r2_level2: n must be positive")
    t1 = e1 / nn + u1
    t0 = e0 / nn + u0
    if t0 <= 0.0:
        raise ValueError("snijders_bosker_r2_level2: baseline total variance must be positive")
    return RichResult(payload={
        "estimate": float(1.0 - t1 / t0), "total1": t1, "total0": t0,
        "n": nn,
        "limit_large_n": float(1.0 - u1 / u0) if u0 > 0.0 else float("nan"),
        "reduction": float(t0 - t1),
        "method": "R2_2 = 1 - (s2_e1/n+s2_u1)/(s2_e0/n+s2_u0) [Snijders & Bosker 1994]"})


# CANONICAL TEST
# >>> from .snr2 import snijders_bosker_r2_level1 as r1
# >>> # n = 1 reproduces the level-1 measure exactly
# >>> a = snijders_bosker_r2_level2(1.0, 0.5, 4.0, 1.0, 1.0)["estimate"]
# >>> assert abs(a - r1(1.0, 0.5, 4.0, 1.0)["estimate"]) < 1e-15
# >>> # identical to the baseline: exactly zero
# >>> assert abs(snijders_bosker_r2_level2(2.0, 1.0, 2.0, 1.0, 7.0)["estimate"]) < 1e-15


def cheatsheet():
    return "snr2u(s2_e1, s2_u1, s2_e0, s2_u0, n): Snijders-Bosker level-2 R^2."
