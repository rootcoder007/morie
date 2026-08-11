# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rao-Scott corrections to Pearson chi-square for complex surveys."""

from . import _array_core as np
from . import _sci_core as sci

from ._richresult import RichResult

__all__ = ["raoscot", "rao_scott_chisq"]


def _chisq_sf(x, df):
    # Survival function of chi-square_df at x via the regularized
    # upper incomplete gamma function Q(df/2, x/2).
    if x <= 0:
        return 1.0
    return float(sci.gammaincc(df / 2.0, x / 2.0))


def raoscot(X2, df, deltas, kappa=None):
    """
    First- and second-order Rao-Scott corrected chi-square tests.

    Let X2 be the Pearson statistic that would have nu degrees of
    freedom under simple random sampling, and let delta_1..delta_nu be
    the generalized design effects (eigenvalue-type quantities of Rao
    and Scott 1981). With dbar = mean(delta) and squared coefficient
    of variation c^2 = sum (delta_l - dbar)^2 / (nu dbar^2):

        first order:   X2_RS1 = X2 / dbar            ~ chi^2_nu
        second order:  X2_RS2 = X2 / (dbar (1 + c^2)) ~ chi^2_{nu/(1+c^2)}

    and the Thomas-Rao (1987) F version

        F_TR = X2 / (nu dbar)  ~  F_{nu/(1+c^2), kappa nu/(1+c^2)}

    where kappa is the degrees of freedom of the design variance
    estimator (if supplied).

    Parameters
    ----------
    X2 : float
        Pearson chi-square statistic computed from the weighted table.
    df : int
        Simple-random-sampling degrees of freedom nu.
    deltas : array-like
        Generalized design effects delta_1..delta_nu. A scalar is
        treated as a common design effect d0 (then c^2 = 0 and the
        two corrections coincide).
    kappa : float, optional
        Degrees of freedom of the variance estimator; enables the
        Thomas-Rao F p-value.

    Returns
    -------
    result : RichResult
        Keys: estimate (X2_RS2), rs1, rs2, f_tr, df, df2, dbar, c2,
        p_rs1, p_rs2, p_f, method.

    References
    ----------
    Rao, J. N. K. and Scott, A. J. (1981), "The analysis of categorical
    data from complex sample surveys: chi-squared tests for goodness of
    fit and independence in two-way tables", JASA 76(374), 221-230.
    Thomas, D. R. and Rao, J. N. K. (1987), "Small-sample comparisons of
    level and power for simple goodness-of-fit statistics under cluster
    sampling", JASA 82(398), 630-636. Formulas as printed in Bilder,
    C. R. and Loughin, T. M. (2014), "Analysis of Categorical Data
    with R", CRC Press, sec. 6.3.5 "Tests of independence: Rao-Scott
    methods" (X2_RS1 = X2/dbar; X2_RS2 = X2/[dbar(1+c^2)] with
    chi^2_{nu/(1+c^2)}; eq. 6.12 F_TR) [local source:
    library/pdf/Analysis of Categorical Data with R ... BILDER.pdf,
    pp. 469-470]. P-values anchored against base R pchisq/pf.
    """
    X2 = float(X2)
    nu = float(df)
    if X2 < 0:
        raise ValueError("X2 must be nonnegative")
    if nu < 1:
        raise ValueError("df must be >= 1")
    d = np.atleast_1d(np.asarray(deltas, dtype=float))
    if len(d) == 1:
        dbar = float(d[0])
        c2 = 0.0
    else:
        if len(d) != int(nu):
            raise ValueError("need one generalized deff per degree of freedom")
        dbar = float(np.mean(d))
        c2 = float(np.sum((d - dbar) ** 2)) / (nu * dbar * dbar)
    if dbar <= 0:
        raise ValueError("design effects must be positive")
    rs1 = X2 / dbar
    rs2 = X2 / (dbar * (1.0 + c2))
    df2 = nu / (1.0 + c2)
    p_rs1 = _chisq_sf(rs1, nu)
    p_rs2 = _chisq_sf(rs2, df2)
    f_tr = X2 / (nu * dbar)
    if kappa is None:
        p_f = float("nan")
        ddf = float("nan")
    else:
        ddf = float(kappa) * df2
        # F survival function via the regularized incomplete beta:
        # P(F > f) = I_{ddf/(ddf + ndf f)}(ddf/2, ndf/2).
        z = ddf / (ddf + df2 * f_tr)
        p_f = float(sci.betainc(ddf / 2.0, df2 / 2.0, z))
    return RichResult(
        payload={
            "estimate": rs2,
            "rs1": rs1,
            "rs2": rs2,
            "f_tr": f_tr,
            "df": nu,
            "df2": df2,
            "ddf": ddf,
            "dbar": dbar,
            "c2": c2,
            "p_rs1": p_rs1,
            "p_rs2": p_rs2,
            "p_f": p_f,
            "method": "Rao-Scott corrected chi-square (first/second order + Thomas-Rao F)",
        }
    )


def rao_scott_chisq(X2, df, deltas, kappa=None):
    return raoscot(X2, df, deltas, kappa=kappa)


def cheatsheet():
    return "raoscot: Rao-Scott first/second-order corrected chi-square (Bilder-Loughin sec 6.3.5)"
