"""Centralization indices ACE and RCE (Duncan 1961; Massey and Denton 1988).

Massey, D. S. and Denton, N. A. (1988). The dimensions of residential segregation. Social
Forces 67, 281-315.
"""

from ._richresult import RichResult
from ._segcore import col_totals, counts, ssum

__all__ = ["centralization_index"]


def centralization_index(x, center_distance, area=None):
    r"""Centralization: how close a group lives to the city centre.

    Units are ordered by distance to the centre; with cumulative shares X_i (group k),
    Y_i (group l) and A_i (area),
    ACE_k = sum_i X_{i-1} A_i - sum_i X_i A_{i-1} and
    RCE_kl = sum_i X_{i-1} Y_i - sum_i X_i Y_{i-1} (units at equal distance pooled first).

    Parameters
    ----------
    x : units x groups counts
    center_distance : sequence
        Distance of each unit to the centre.
    area : sequence, optional
        Needed for ACE.

    Returns
    -------
    RichResult
        Keys: RCE (groups x groups), ACE (per group, with ``area``).

    References
    ----------
    Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.

    Examples
    --------
    >>> round(centralization_index([[10, 0], [5, 5], [0, 10]], [0, 1, 2])["RCE"][0][1], 12)
    0.888888888889
    """
    keep = [i for i, r in enumerate(x) if ssum(float(v) for v in r) > 0]
    X = counts(x)
    n, g = len(X), len(X[0])
    tot = col_totals(X)
    dc = [float(center_distance[i]) for i in keep]
    order = sorted(range(n), key=lambda i: (dc[i], i))
    out = {}
    if area is not None:
        a = [float(area[i]) for i in keep]
        A = ssum(a)
        ACE = []
        for k in range(g):
            cx, ca, XI, AI = 0.0, 0.0, [], []
            for i in order:
                cx += X[i][k]
                ca += a[i]
                XI.append(cx / tot[k])
                AI.append(ca / A)
            ACE.append(ssum(XI[i - 1] * AI[i] for i in range(1, n)) - ssum(XI[i] * AI[i - 1] for i in range(1, n)))
        out["ACE"] = ACE
    levels = sorted(set(dc))
    pooled = [[ssum(X[i][k] for i in range(n) if dc[i] == lv) for k in range(g)] for lv in levels]
    cum = []
    for k in range(g):
        s, c = 0.0, []
        for row in pooled:
            s += row[k]
            c.append(s / tot[k])
        cum.append(c)
    m = len(levels)
    out["RCE"] = [
        [
            ssum(cum[p][i - 1] * cum[q][i] for i in range(1, m)) - ssum(cum[p][i] * cum[q][i - 1] for i in range(1, m))
            for q in range(g)
        ]
        for p in range(g)
    ]
    return RichResult(title="Centralization indices", summary_lines=[("groups", g)], payload=out)


def cheatsheet():
    return "centidx: Massey-Denton centralization ACE and RCE"
