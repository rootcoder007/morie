"""Rubin's rules for combining multiple-imputation estimates (Rubin 1987)."""

import math

from ._richresult import RichResult

__all__ = ["miefcl", "mi_rubin_rules"]


def miefcl(estimates, variances, nu_com=None):
    """
    Pool a scalar estimate across multiply imputed data sets.

    Rubin's rules (Rubin 1987, Sec. 3.3; van Buuren 2018, Eqs.
    2.16-2.32): with m complete-data estimates Q_l and their
    variances U_l,

        Qbar = mean(Q_l),
        Ubar = mean(U_l)                (within variance),
        B = var(Q_l)  (denominator m-1) (between variance),
        T = Ubar + (1 + 1/m) B          (total variance).

    Also returned: relative increase in variance
    r = (1 + 1/m) B / Ubar, proportion of variance attributable to
    missingness lambda = (1 + 1/m) B / T (Eq. 2.24), fraction of
    missing information fmi = (r + 2/(nu+3)) / (1 + r) (Eq. 2.26 form),
    and the degrees of freedom: nu_old = (m - 1) / lambda^2
    (Eq. 2.30), replaced by the Barnard-Rubin small-sample value when
    ``nu_com`` is given (Eqs. 2.31-2.32; see also morie.fn.midegf).

    Sources
    -------
    Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in
    Surveys*. Wiley, Sec. 3.3 (the combining rules).
    van Buuren, S. (2018). *Flexible Imputation of Missing Data*,
    2nd ed., Chapman & Hall/CRC, Sec. 2.3, Eqs. 2.16-2.32 (local
    copy fetched-wave3/vanbuuren-fimd-ch2-rubins-rules.html).
    Barnard, J. & Rubin, D. B. (1999). Small-sample degrees of
    freedom with multiple imputation. *Biometrika*, 86, 948-955.

    Parameters
    ----------
    estimates : sequence of float
        Complete-data point estimates, one per imputation.
    variances : sequence of float
        Complete-data variances (squared standard errors).
    nu_com : float, optional
        Complete-data degrees of freedom (e.g. n - k); enables the
        Barnard-Rubin adjusted df.

    Returns
    -------
    RichResult
        Keys: estimate, se, t (total variance), ubar, b, m, riv,
        lambda_, fmi, df.
    """
    q = [float(v) for v in estimates]
    u = [float(v) for v in variances]
    m = len(q)
    if m < 2:
        raise ValueError("need at least two imputations")
    if len(u) != m:
        raise ValueError("estimates and variances must have equal length")
    if any(v < 0 for v in u):
        raise ValueError("variances must be non-negative")
    qbar = sum(q) / m
    ubar = sum(u) / m
    b = sum((x - qbar) ** 2 for x in q) / (m - 1)
    t = ubar + (1.0 + 1.0 / m) * b
    if t <= 0:
        raise ValueError("total variance is not positive")
    lam = (b + b / m) / t
    riv = float("inf") if ubar == 0 else (1.0 + 1.0 / m) * b / ubar
    if lam <= 0:
        df_old = float("inf")
    else:
        df_old = (m - 1) / lam ** 2
    if nu_com is None:
        df = df_old
    else:
        nc = float(nu_com)
        if nc <= 0:
            raise ValueError("nu_com must be positive")
        nu_obs = (nc + 1.0) / (nc + 3.0) * nc * (1.0 - lam)
        if math.isinf(df_old):
            df = min(nu_obs, nc)
        else:
            df = df_old * nu_obs / (df_old + nu_obs)
    fmi = (riv + 2.0 / (df + 3.0)) / (1.0 + riv) \
        if not math.isinf(riv) else 1.0
    return RichResult(payload={
        "estimate": qbar,
        "se": math.sqrt(t),
        "t": t,
        "ubar": ubar,
        "b": b,
        "m": m,
        "riv": riv,
        "lambda_": lam,
        "fmi": fmi,
        "df": df,
        "method": "Rubin's rules (Rubin 1987; van Buuren 2018 Sec. 2.3)",
    })


# long descriptive alias (stub-era name)
mi_rubin_rules = miefcl


def cheatsheet():
    return "miefcl: pool MI estimates, T = Ubar + (1+1/m)B, Barnard-Rubin df"
