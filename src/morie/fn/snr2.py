# morie.fn -- function file (rootcoder007/morie)
"""Snijders-Bosker level-1 explained variance for a two-level model."""

from ._richresult import RichResult

__all__ = ["snijders_bosker_r2_level1"]


def snijders_bosker_r2_level1(sigma2_e1, sigma2_u1, sigma2_e0, sigma2_u0):
    """Proportion of level-1 variance explained, relative to an empty model.

    Snijders and Bosker's point is that the naive multilevel analogue of
    R-squared can go *negative* when predictors are added, because the
    separate variance components are not each individually reduced.  The
    fix is to define explained variance on the total residual variance of
    an individual observation, ``sigma2_e + sigma2_u``, comparing the
    fitted model against the empty (intercept-only) model.  So defined it
    behaves: adding a useless predictor leaves it near zero rather than
    driving it below.

    Formula: ``R2_1 = 1 - (sigma2_e1 + sigma2_u1) / (sigma2_e0 + sigma2_u0)``.

    Parameters
    ----------
    sigma2_e1, sigma2_u1 : float
        Level-1 (residual) and level-2 (intercept) variance components of
        the fitted model.  Non-negative.
    sigma2_e0, sigma2_u0 : float
        The same two components of the empty baseline model.  Their sum
        must be positive.

    Returns
    -------
    RichResult
        ``estimate`` (R2_1), ``total1``, ``total0``, ``icc0``, ``icc1``,
        ``reduction`` (absolute variance removed), ``method``.

    References
    ----------
    Snijders, T. A. B. & Bosker, R. J. (1994).  Modeled variance in
    two-level models.  Sociological Methods & Research 22(3):342-363.
    <https://doi.org/10.1177/0049124194022003004>
    Snijders, T. A. B. & Bosker, R. J. (2012).  Multilevel Analysis, 2nd
    edition, Sage, section 7.2.
    """
    e1, u1, e0, u0 = float(sigma2_e1), float(sigma2_u1), float(sigma2_e0), float(sigma2_u0)
    for v in (e1, u1, e0, u0):
        if v < 0.0:
            raise ValueError("snijders_bosker_r2_level1: variance components must be non-negative")
    t1 = e1 + u1
    t0 = e0 + u0
    if t0 <= 0.0:
        raise ValueError("snijders_bosker_r2_level1: baseline total variance must be positive")
    return RichResult(payload={
        "estimate": float(1.0 - t1 / t0), "total1": t1, "total0": t0,
        "icc0": float(u0 / t0), "icc1": float(u1 / t1) if t1 > 0.0 else 0.0,
        "reduction": float(t0 - t1),
        "method": "R2_1 = 1 - (s2_e1+s2_u1)/(s2_e0+s2_u0) [Snijders & Bosker 1994]"})


# CANONICAL TEST
# >>> # identical to the baseline: exactly zero explained
# >>> assert abs(snijders_bosker_r2_level1(2.0, 1.0, 2.0, 1.0)["estimate"]) < 1e-15
# >>> # all residual variance removed: exactly one
# >>> assert abs(snijders_bosker_r2_level1(0.0, 0.0, 2.0, 1.0)["estimate"] - 1.0) < 1e-15
# >>> # no level-2 variance: collapses to the single-level R^2
# >>> assert abs(snijders_bosker_r2_level1(1.0, 0.0, 4.0, 0.0)["estimate"] - 0.75) < 1e-15


def cheatsheet():
    return "snr2(s2_e1, s2_u1, s2_e0, s2_u0): Snijders-Bosker level-1 R^2."
