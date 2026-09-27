"""Scheffe interval for a linear contrast of means.

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eqs (12.7)-(12.9).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qf

__all__ = ["scheffeci"]


def scheffeci(groups, contrast, alpha=0.05):
    """Psi = sum c_l mu_l (12.7), estimated by sum c_l Xbar_l, with interval Psi_hat +- S (12.9).

    S = sqrt((L - 1) f_{1-alpha} MSWG sum c_l^2 / n_l), f the F quantile on
    (L - 1, N - L) degrees of freedom; the intervals hold simultaneously over
    all contrasts.

    Parameters
    ----------
    groups : list of L sequences
    contrast : sequence of L floats
    alpha : float

    Returns
    -------
    RichResult
        Keys: estimate, S, ci, mswg, df.

    References
    ----------
    Scheffe, H. (1953). Biometrika 40, 87-104.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eqs (12.7)-(12.9).

    Examples
    --------
    >>> scheffeci([[1, 2, 3], [2, 3, 4], [6, 7, 8]], [1, -1, 0])["estimate"]
    -1.0
    """
    gs = [[float(v) for v in g] for g in groups]
    c = [float(v) for v in contrast]
    L = len(gs)
    N = sum(len(g) for g in gs)
    if L < 2 or len(c) != L or N - L < 1:
        raise ValueError("need L >= 2 groups, one coefficient per group and N > L")
    means = [sum(g) / len(g) for g in gs]
    df = N - L
    mswg = sum(sum((v - m) ** 2 for v in g) for g, m in zip(gs, means)) / df
    est = sum(a * m for a, m in zip(c, means))
    S = math.sqrt((L - 1) * qf(1 - alpha, L - 1, df) * mswg * sum(a * a / len(g) for a, g in zip(c, gs)))
    return RichResult(
        title="Scheffe contrast interval",
        summary_lines=[("estimate", est), ("S", S)],
        payload={"estimate": est, "S": S, "ci": (est - S, est + S), "mswg": mswg, "df": df},
    )


def cheatsheet():
    return "scheffeci: Scheffe interval Psi_hat +- sqrt((L-1) f MSWG sum c^2/n). Wilcox (2017) eqs (12.7)-(12.9)."
