"""Evenness indices: Theil's entropy H, Atkinson and Gini (Massey and Denton 1988).

Theil, H. and Finizza, A. J. (1971). A note on the measurement of racial integration of
schools by means of informational concepts. Journal of Mathematical Sociology 1, 187-194.
Atkinson, A. B. (1970). On the measurement of inequality. Journal of Economic Theory 2,
244-263. Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.
"""

import math

from ._richresult import RichResult
from ._segcore import col_totals, counts, row_totals, ssum

__all__ = ["segregation_evenness"]


def _ent(p):
    return (p * math.log(1 / p) if p > 0 else 0.0) + ((1 - p) * math.log(1 / (1 - p)) if p < 1 else 0.0)


def segregation_evenness(x, delta=0.5):
    r"""Evenness of each group against the rest: Theil's entropy index
    H_k = sum_i t_i (E_k - E_ik) / (E_k T), with E the binary entropy of the group share;
    Atkinson A_k = 1 - P/(1 - P) (sum_i (1 - p_i)^(1-b) p_i^b t_i / (P T))^(1/(1-b)), b = delta;
    and the Gini index G_k = sum_ij t_i t_j |p_i - p_j| / (2 T^2 P (1 - P)).

    Parameters
    ----------
    x : units x groups counts
    delta : float
        Atkinson shape parameter b in (0, 1).

    Returns
    -------
    RichResult
        Keys: H, atkinson, gini (per group).

    References
    ----------
    Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.
    Atkinson, A. B. (1970). Journal of Economic Theory 2, 244-263.

    Examples
    --------
    >>> segregation_evenness([[10, 0], [0, 10]])["H"]
    [1.0, 1.0]
    """
    X = counts(x)
    n, g = len(X), len(X[0])
    tot, t = col_totals(X), row_totals(X)
    T = ssum(t)
    H, AT, G = [], [], []
    for k in range(g):
        P = tot[k] / T
        p = [X[i][k] / t[i] for i in range(n)]
        Ek = _ent(P)
        H.append(ssum(t[i] * (Ek - _ent(p[i])) for i in range(n)) / (Ek * T))
        s = ssum((1 - p[i]) ** (1 - delta) * p[i] ** delta * t[i] / (P * T) for i in range(n))
        AT.append(1 - P / (1 - P) * s ** (1 / (1 - delta)))
        G.append(ssum(t[i] * t[j] * abs(p[i] - p[j]) for i in range(n) for j in range(n)) / (2 * T * T * P * (1 - P)))
    return RichResult(
        title="Evenness indices", summary_lines=[("groups", g)], payload={"H": H, "atkinson": AT, "gini": G}
    )


def cheatsheet():
    return "evenidx: Theil entropy H, Atkinson and Gini segregation indices"
