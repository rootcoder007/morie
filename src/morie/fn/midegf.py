# SPDX-License-Identifier: AGPL-3.0-or-later
"""Barnard-Rubin degrees of freedom for multiple imputation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["midegf", "mi_degrees_of_freedom"]


def midegf(B, W, m, dfcom=None):
    """
    Degrees of freedom for multiple-imputation inference.

    With between variance B, within variance W (Ubar) and m
    imputations, total variance T = W + (1 + 1/m) B and

        lambda  = (1 + 1/m) B / T
        nu_old  = (m - 1) / lambda^2               (Rubin 1987)

    Barnard and Rubin (1999) adjust for a finite complete-data
    degrees-of-freedom nu_com:

        nu_obs = nu_com (nu_com + 1)(1 - lambda) / (nu_com + 3)
        nu_adj = nu_old * nu_obs / (nu_old + nu_obs)

    which is never larger than nu_com (Rubin's nu_old can exceed it).

    Parameters
    ----------
    B : float
        Between-imputation variance.
    W : float
        Within-imputation variance (Ubar).
    m : int
        Number of imputations (m >= 2).
    dfcom : float, optional
        Complete-data degrees of freedom nu_com. If omitted, the
        Rubin (1987) large-sample nu_old is returned.

    Returns
    -------
    result : RichResult
        Keys: estimate (the df), df_old, lambda, m, dfcom, method.

    References
    ----------
    Barnard, J. and Rubin, D. B. (1999), "Small-sample degrees of
    freedom with multiple imputation", Biometrika 86(4), 948-955
    (nu_adj = (nu_old^-1 + nu_obs^-1)^-1). Rubin, D. B. (1987),
    "Multiple Imputation for Nonresponse in Surveys", Wiley, ch. 3.
    Formulas as printed in van Buuren, S. (2018), "Flexible Imputation
    of Missing Data", 2nd ed., sec. 2.3.6, eqs. (2.30)-(2.32) [source
    snapshot: library/pdf/fetched-wave3/fimd-whyandwhen.html].
    Anchored against mice:::barnard.rubin (amices/mice,
    R/barnard.rubin.R, snapshot
    library/pdf/fetched-wave3/mice-barnard-rubin-source.R).
    """
    B = float(B)
    W = float(W)
    m = int(m)
    if m < 2:
        raise ValueError("need m >= 2 imputations")
    if B < 0 or W < 0:
        raise ValueError("variances must be nonnegative")
    t = W + (1.0 + 1.0 / m) * B
    lam = (1.0 + 1.0 / m) * B / t if t > 0 else float("nan")
    if lam < 1e-12:
        df_old = float("inf")
    else:
        df_old = (m - 1.0) / (lam * lam)
    if dfcom is None:
        df = df_old
        nu_obs = float("nan")
    else:
        dfcom = float(dfcom)
        nu_obs = dfcom * (dfcom + 1.0) * (1.0 - lam) / (dfcom + 3.0)
        if df_old == float("inf"):
            df = nu_obs
        else:
            df = df_old * nu_obs / (df_old + nu_obs)
    return RichResult(
        payload={
            "estimate": df,
            "df_old": df_old,
            "nu_obs": nu_obs,
            "lambda": lam,
            "m": m,
            "dfcom": float("nan") if dfcom is None else dfcom,
            "method": "Barnard-Rubin MI degrees of freedom",
        }
    )


def mi_degrees_of_freedom(B, W, m, dfcom=None):
    return midegf(B, W, m, dfcom=dfcom)


def cheatsheet():
    return "midegf: MI degrees of freedom (Rubin 1987; Barnard-Rubin 1999)"
