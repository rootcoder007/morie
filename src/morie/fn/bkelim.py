"""Backward elimination of regressors by the partial F statistic (F-out rule)."""

from ._richresult import RichResult
from .linsys import _householder_ls

__all__ = ["backward_elimination"]


def backward_elimination(X, y, names=None, f_out=4.0):
    r"""Drop regressors one at a time while the smallest partial F is below ``f_out``.

    For the current model with p regressors and an intercept, each
    regressor's statistic is (Hedderich, Sachs & Reynarowych 2023, eq 8.39;
    ``drop1(fit, test = "F")``)

    .. math::  \hat F = \frac{RSS_{(p-1)} - RSS_{(p)}}{RSS_{(p)}/(n-(p+1))},

    and the one with the smallest :math:`\hat F < F_{out}` is removed.

    Parameters
    ----------
    X : n x p nested sequence
        Regressors (the intercept is added).
    y : sequence of n floats
    names : sequence of p str, optional
    f_out : float

    Returns
    -------
    RichResult
        ``selected`` (names kept), ``removed`` (list of (name, F)),
        ``coefficients`` (intercept first), ``rss``, ``last_f`` (F of the
        kept regressors at the final step).

    References
    ----------
    Draper, N. R. & Smith, H. (1998). Applied Regression Analysis (3rd ed.),
    sec. 15.2.
    """
    rows = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    n, p = len(rows), len(rows[0])
    names = [f"x{j + 1}" for j in range(p)] if names is None else list(names)
    if n != len(yy) or len(names) != p:
        raise ValueError("X must be n x p with n = len(y) and p names")

    def fit(cols):
        return _householder_ls([[1.0] + [r[j] for j in cols] for r in rows], yy)

    active, removed, last = list(range(p)), [], {}
    while True:
        coef, rss = fit(active)
        if not active:
            break
        df = n - (len(active) + 1)
        if df <= 0:
            raise ValueError("too few observations for the full model")
        fs = {j: (fit([k for k in active if k != j])[1] - rss) / (rss / df) for j in active}
        last = {names[j]: fs[j] for j in active}
        jmin = min(active, key=lambda j: (fs[j], j))
        if fs[jmin] >= f_out:
            break
        removed.append((names[jmin], fs[jmin]))
        active.remove(jmin)
    return RichResult(
        title="Backward elimination (F-out)",
        summary_lines=[("selected", [names[j] for j in active]), ("rss", rss)],
        payload={
            "selected": [names[j] for j in active],
            "removed": removed,
            "coefficients": coef,
            "rss": rss,
            "last_f": last,
        },
    )


def cheatsheet():
    return "bkelim: remove argmin F = (RSS_{p-1} - RSS_p)/(RSS_p/(n - p - 1)) while F < F_out"
