"""Rao-Scott first- and second-order corrections and the Thomas-Rao F test for Pearson/LRT statistics.

Bilder & Loughin (2025), Analysis of Categorical Data with R, Sec 6.3, eq (6.12).
"""

from ._richresult import RichResult
from ._rrng_core import pchisq, pf

__all__ = ["raoscott"]


def raoscott(statistic, deltas, kappa=None):
    r"""Refer X^2 (or an LRT statistic) to the distribution implied by generalized design effects.

    With nu = len(deltas), d_bar = mean(delta) and
    c^2 = sum (delta_l - d_bar)^2 / (nu d_bar^2):
    first order X^2/d_bar ~ chi2_nu; second order X^2/[d_bar(1 + c^2)] ~
    chi2_{nu/(1+c^2)}; Thomas-Rao F_TR = X^2/(nu d_bar) (6.12) ~
    F_{nu/(1+c^2), kappa nu/(1+c^2)} with kappa the degrees of freedom of the
    variance estimate. Since d_bar^2 (1 + c^2) = sum(delta^2)/nu, the second-order
    statistic is the Satterthwaite form nu d_bar X^2 / sum(delta^2) on
    (sum delta)^2 / sum(delta^2) df (MRCV's I J X^2 / sum(lambda^2) is the case d_bar = 1).

    Parameters
    ----------
    statistic : float
    deltas : sequence of float
        Generalized design effects (eigenvalues).
    kappa : float, optional
        Variance-estimate degrees of freedom for the F test.

    Returns
    -------
    RichResult
        Keys: rs1, rs1_df, rs1_p, rs2, rs2_df, rs2_p, f_tr, f_df, f_p, d_bar, c2.

    References
    ----------
    Rao, J. N. K. & Scott, A. J. (1981). JASA 76, 221-230.
    Thomas, D. R. & Rao, J. N. K. (1987). JASA 82, 630-636.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Sec 6.3, eq (6.12).

    Examples
    --------
    >>> raoscott(10.0, [2.0, 2.0])["rs1"]
    5.0
    """
    d = [float(v) for v in deltas]
    nu = len(d)
    if nu < 1 or min(d) <= 0 or statistic < 0:
        raise ValueError("need positive design effects and a non-negative statistic")
    db = sum(d) / nu
    c2 = sum((v - db) ** 2 for v in d) / (nu * db * db)
    rs1 = statistic / db
    rs2 = statistic / (db * (1 + c2))
    df2 = nu / (1 + c2)
    ftr = statistic / (nu * db)
    fdf = (df2, kappa * nu / (1 + c2)) if kappa is not None else None
    return RichResult(
        title="Rao-Scott corrections",
        summary_lines=[("RS1", rs1), ("RS2", rs2), ("df2", df2)],
        payload={
            "rs1": rs1,
            "rs1_df": nu,
            "rs1_p": pchisq(rs1, nu, lower_tail=False),
            "rs2": rs2,
            "rs2_df": df2,
            "rs2_p": pchisq(rs2, df2, lower_tail=False),
            "f_tr": ftr,
            "f_df": fdf,
            "f_p": pf(ftr, fdf[0], fdf[1], lower_tail=False) if fdf else None,
            "d_bar": db,
            "c2": c2,
        },
    )


def cheatsheet():
    return (
        "raoscott: Rao-Scott RS1/RS2 and Thomas-Rao F corrections from generalized deffs. Bilder & Loughin eq (6.12)."
    )
