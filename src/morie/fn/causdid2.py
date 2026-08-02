# morie.fn -- function file (rootcoder007/morie)
"""Canonical 2x2 difference-in-differences."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["causal_did_2x2"]


def causal_did_2x2(y, treated, post):
    r"""The canonical two-group, two-period DiD.

    .. math:: \widehat{\mathrm{ATT}}
              = (\bar y_{T,1} - \bar y_{T,0})
              - (\bar y_{C,1} - \bar y_{C,0}),

    computed both from the four cell means and as the interaction
    coefficient of the saturated OLS
    :math:`y = \beta_0 + \beta_1 T + \beta_2 P + \beta_3 (T \times P)`,
    whose identity with the cell-mean contrast is the standard check.
    The OLS route supplies the classical SE and p-value.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome (pooled cross sections or stacked panel).
    treated : array-like of {0, 1}, shape (n,)
        Group indicator.
    post : array-like of {0, 1}, shape (n,)
        Period indicator.

    Returns
    -------
    RichResult
        keys: ``att``, ``se``, ``p_value``, ``cell_means`` dict with
        keys ``T0 T1 C0 C1``, ``n``, ``method``.

    References
    ----------
    Card, D. & Krueger, A. B. (1994). Minimum wages and employment: a
    case study of the fast-food industry in New Jersey and
    Pennsylvania. *American Economic Review*, 84(4), 772-793. (the
    canonical 2x2 application)
    """
    y = np.asarray(y, dtype=float).ravel()
    T = np.asarray(treated, dtype=float).ravel()
    P = np.asarray(post, dtype=float).ravel()
    n = y.size
    if T.size != n or P.size != n:
        raise ValueError("y, treated, post must have equal length.")
    for v, name in ((T, "treated"), (P, "post")):
        if not np.all(np.isin(v, (0.0, 1.0))):
            raise ValueError(f"{name} must be binary 0/1.")
    cells = {}
    for gname, g in (("T", T == 1), ("C", T == 0)):
        for pname, p in (("1", P == 1), ("0", P == 0)):
            m = g & p
            if m.sum() == 0:
                raise ValueError(f"empty cell {gname}{pname}: all four 2x2 cells need data.")
            cells[gname + pname] = float(y[m].mean())

    att = (cells["T1"] - cells["T0"]) - (cells["C1"] - cells["C0"])

    D = np.column_stack([np.ones(n), T, P, T * P])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    dof = n - 4
    s2 = float((resid**2).sum() / dof) if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(D.T @ D)
    se = float(np.sqrt(cov[3, 3]))
    p = float(2 * stats.t.sf(abs(beta[3]) / se, dof)) if dof > 0 and se > 0 else float("nan")

    return RichResult(
        payload={
            "att": float(att),
            "se": se,
            "p_value": p,
            "cell_means": cells,
            "n": int(n),
            "method": "Canonical 2x2 difference-in-differences",
        }
    )


def cheatsheet():
    return "causdid2: ATT = (T1-T0) - (C1-C0); OLS interaction gives the SE"
