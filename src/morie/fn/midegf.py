"""Barnard-Rubin small-sample MI degrees of freedom (Barnard & Rubin 1999)."""

import math

from ._richresult import RichResult

__all__ = ["midegf", "mi_degrees_of_freedom"]


def midegf(b, t, m, nu_com=None):
    """
    Degrees of freedom for multiple-imputation inference.

    With between variance B, total variance T and m imputations,
    lambda = (1 + 1/m) B / T is the proportion of variance
    attributable to the missing data (van Buuren 2018, Eq. 2.24).
    The classical df is nu_old = (m - 1) / lambda^2 (Eq. 2.30;
    Rubin 1987, Eq. 3.3.7 form).  Barnard & Rubin (1999) noted
    nu_old can exceed the complete-data sample size, "clearly
    inappropriate", and proposed the small-sample adjustment
    (Eqs. 2.31-2.32):

        nu_obs = ((nu_com + 1) / (nu_com + 3)) * nu_com * (1 - lambda)
        nu = nu_old * nu_obs / (nu_old + nu_obs),

    which is always <= nu_com, reduces to nu_old when nu_com -> inf,
    equals nu_com at lambda = 0 and 0 at lambda = 1.

    Sources
    -------
    Barnard, J. & Rubin, D. B. (1999). Small-sample degrees of
    freedom with multiple imputation. *Biometrika*, 86, 948-955.
    van Buuren, S. (2018). *Flexible Imputation of Missing Data*,
    2nd ed., Sec. 2.3.6, Eqs. 2.30-2.32 (local copy
    fetched-wave3/vanbuuren-fimd-ch2-rubins-rules.html; the mice
    implementation is kept test-only at
    fetched-wave3/mice-barnard-rubin-source.R).

    Parameters
    ----------
    b : float
        Between-imputation variance.
    t : float
        Total variance T = Ubar + (1 + 1/m) B.
    m : int
        Number of imputations (>= 2).
    nu_com : float, optional
        Complete-data degrees of freedom; when omitted the classical
        nu_old is returned.

    Returns
    -------
    RichResult
        Keys: df, df_old, nu_obs (None without nu_com), lambda_.
    """
    b = float(b)
    t = float(t)
    m = int(m)
    if m < 2:
        raise ValueError("m must be at least 2")
    if b < 0 or t <= 0:
        raise ValueError("need b >= 0 and t > 0")
    lam = (1.0 + 1.0 / m) * b / t
    lam = min(max(lam, 0.0), 1.0)
    df_old = float("inf") if lam == 0 else (m - 1) / lam ** 2
    nu_obs = None
    if nu_com is None:
        df = df_old
    else:
        nc = float(nu_com)
        if nc <= 0:
            raise ValueError("nu_com must be positive")
        nu_obs = (nc + 1.0) / (nc + 3.0) * nc * (1.0 - lam)
        if math.isinf(df_old):
            df = nu_obs
        else:
            df = df_old * nu_obs / (df_old + nu_obs) \
                if df_old + nu_obs > 0 else 0.0
    return RichResult(payload={
        "df": df,
        "df_old": df_old,
        "nu_obs": nu_obs,
        "lambda_": lam,
        "m": m,
        "method": "Barnard-Rubin (1999) adjusted MI df",
    })


# long descriptive alias (stub-era name)
mi_degrees_of_freedom = midegf


def cheatsheet():
    return "midegf: nu = nu_old*nu_obs/(nu_old+nu_obs), Barnard-Rubin 1999"
