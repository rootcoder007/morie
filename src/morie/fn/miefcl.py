# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rubin's rules for combining multiple-imputation estimates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["miefcl", "multiple_imputation_combine"]


def miefcl(estimates, ses, dfcom=None):
    """
    Combine m completed-data estimates by Rubin's rules.

    For a scalar estimand Q with completed-data estimates Q_i and
    variances U_i (i = 1..m):

        Qbar = (1/m) sum Q_i
        Ubar = (1/m) sum U_i                       (within variance)
        B    = (1/(m-1)) sum (Q_i - Qbar)^2        (between variance)
        T    = Ubar + (1 + 1/m) B                  (total variance)

    Inference uses t_nu with

        r      = (1 + 1/m) B / Ubar                (rel. incr. variance)
        lambda = (1 + 1/m) B / T
        nu_old = (m - 1) / lambda^2                (Rubin 1987)
        gamma  = (r + 2/(nu + 3)) / (1 + r)        (fraction missing info)

    If ``dfcom`` (complete-data degrees of freedom) is given, the
    Barnard-Rubin (1999) small-sample adjustment
    nu_adj = nu_old * nu_obs / (nu_old + nu_obs) with
    nu_obs = dfcom (dfcom + 1)(1 - lambda) / (dfcom + 3)
    replaces nu_old.

    Parameters
    ----------
    estimates : array-like, shape (m,)
        Completed-data point estimates Q_i.
    ses : array-like, shape (m,)
        Completed-data standard errors sqrt(U_i).
    dfcom : float, optional
        Complete-data degrees of freedom (Barnard-Rubin adjustment).

    Returns
    -------
    result : RichResult
        Keys: estimate, se, t, ubar, b, m, df, riv, lambda, fmi, method.

    References
    ----------
    Rubin, D. B. (1987), "Multiple Imputation for Nonresponse in
    Surveys", Wiley, New York, ch. 3 (repeated-imputation inference for
    scalar estimands), as printed in van Buuren, S. (2018), "Flexible
    Imputation of Missing Data", 2nd ed., CRC Press, sec. 2.3,
    eqs. (2.17)-(2.32) [source snapshot:
    library/pdf/fetched-wave3/fimd-whyandwhen.html].
    Barnard, J. and Rubin, D. B. (1999), "Small-sample degrees of
    freedom with multiple imputation", Biometrika 86(4), 948-955.
    Anchored against mice::pool (amices/mice, R/pool.R and
    R/barnard.rubin.R).
    """
    q = np.atleast_1d(np.asarray(estimates, dtype=float))
    s = np.atleast_1d(np.asarray(ses, dtype=float))
    m = len(q)
    if len(s) != m:
        raise ValueError("estimates and ses must have equal length")
    if m < 2:
        raise ValueError("need m >= 2 imputations")
    u = s * s
    qbar = float(np.mean(q))
    ubar = float(np.mean(u))
    b = float(np.sum((q - qbar) ** 2)) / (m - 1.0)
    t = ubar + (1.0 + 1.0 / m) * b
    riv = (1.0 + 1.0 / m) * b / ubar if ubar > 0 else float("inf")
    lam = (1.0 + 1.0 / m) * b / t if t > 0 else float("nan")
    if lam < 1e-12:
        df_old = float("inf")
    else:
        df_old = (m - 1.0) / (lam * lam)
    if dfcom is None:
        df = df_old
    else:
        dfcom = float(dfcom)
        nu_obs = dfcom * (dfcom + 1.0) * (1.0 - lam) / (dfcom + 3.0)
        if df_old == float("inf"):
            df = nu_obs
        else:
            df = df_old * nu_obs / (df_old + nu_obs)
    if df == float("inf"):
        fmi = riv / (1.0 + riv)
    else:
        fmi = (riv + 2.0 / (df + 3.0)) / (1.0 + riv)
    return RichResult(
        payload={
            "estimate": qbar,
            "se": float(t) ** 0.5,
            "t": t,
            "ubar": ubar,
            "b": b,
            "m": m,
            "df": df,
            "riv": riv,
            "lambda": lam,
            "fmi": fmi,
            "method": "Rubin's rules MI combination",
        }
    )


multiple_imputation_combine = miefcl


def cheatsheet():
    return "miefcl: Rubin's rules MI combination (Rubin 1987; Barnard-Rubin 1999 df)"
