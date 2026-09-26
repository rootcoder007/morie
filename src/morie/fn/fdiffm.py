"""First-difference model for elongated field layouts (Besag and Kempton)."""

from ._ols_small import ols
from ._richresult import RichResult

__all__ = ["first_difference_model"]


def first_difference_model(z, treatment, column=None, row=None):
    r"""Treatment contrasts from the first-difference model of Besag & Kempton (1986).

    Schabenberger & Gotway (2005, Sec. 6.1.3.3, eqs 6.12-6.13, p. 320): for
    :math:`Z(s) = X\tau + e(s)` with first differences within a column
    homoscedastic and uncorrelated, :math:`\Delta Z = \Delta X\tau +
    \Delta e` is an OLS model. Differencing removes the intercept, so the
    treatment effects are estimated as contrasts with the first level:

    .. math::

        \hat\tau = (X^{*\prime}X^*)^{-1}X^{*\prime}Z^*,\quad
        \widehat{\mathrm{Var}}[\hat\tau] = \hat\sigma^2(X^{*\prime}X^*)^{-1},\quad
        \hat\sigma^2 = \frac{\|Z^* - X^*\hat\tau\|^2}{n - \mathrm{rank}\{X\}}

    (:math:`n - \mathrm{rank}\{X\}` equals the number of differences minus
    the number of contrasts).

    Parameters
    ----------
    z : sequence of float
    treatment : sequence
    column : sequence, optional
        Column of each plot; differences are taken within a column.
    row : sequence of int, optional
        Row of each plot, the order within a column (default: input order).

    Returns
    -------
    RichResult
        ``levels``, ``tau`` (contrasts with the first level), ``se``,
        ``sigma2``, ``df``, ``n_differences``.

    References
    ----------
    Besag, J. & Kempton, R. (1986). Statistical analysis of field
    experiments using neighbouring plots. Biometrics 42, 231-251.
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eqs (6.12)-(6.13), p. 320.
    """
    z = [float(v) for v in z]
    n = len(z)
    column = [0] * n if column is None else list(column)
    row = list(range(n)) if row is None else [int(v) for v in row]
    if not (len(treatment) == len(column) == len(row) == n):
        raise ValueError("`z`, `treatment`, `column` and `row` must be the same length")
    levels = sorted(set(treatment), key=lambda v: (str(type(v)), v))
    if len(levels) < 2:
        raise ValueError("need at least two treatments")
    d = [[1.0 if treatment[i] == lv else 0.0 for lv in levels[1:]] for i in range(n)]
    zs, xs = [], []
    for c in sorted(set(column), key=lambda v: (str(type(v)), v)):
        idx = sorted((i for i in range(n) if column[i] == c), key=lambda i: row[i])
        for a, b in zip(idx, idx[1:]):
            zs.append(z[a] - z[b])
            xs.append([u - v for u, v in zip(d[a], d[b])])
    tau, se, s2, df, _ = ols(xs, zs)
    return RichResult(
        title="First-difference model (Besag and Kempton)",
        summary_lines=[("treatments", len(levels)), ("differences", len(zs)), ("sigma2", s2)],
        payload={"levels": levels, "tau": tau, "se": se, "sigma2": s2, "df": df, "n_differences": len(zs)},
    )


def cheatsheet():
    return "fdiffm: OLS on within-column first differences, eq 6.12-6.13"
