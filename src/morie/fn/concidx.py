"""Concentration indices DEL, ACO and RCO (Hoover 1941; Massey and Denton 1988).

Massey, D. S. and Denton, N. A. (1988). The dimensions of residential segregation. Social
Forces 67, 281-315.
"""

from ._richresult import RichResult
from ._segcore import col_totals, counts, row_totals, ssum

__all__ = ["spatial_concentration"]


def spatial_concentration(x, area):
    r"""Concentration: how little physical space a group occupies.

    DEL_k = 1/2 sum_i |x_ik / X_k - a_i / A| (Hoover's delta). With units sorted by
    increasing area, n1_k is the smallest count whose total population reaches X_k from the
    small end and n2_k the corresponding count from the large end; with
    T1 = sum_{i <= n1} t_i and T2 = sum_{i >= n2} t_i,
    ACO_k = 1 - (sum x_ik a_i / X_k - sum_{i<=n1} t_i a_i / T1) / (sum_{i>=n2} t_i a_i / T2 - sum_{i<=n1} t_i a_i / T1)
    and RCO_kl = ((sum x_ik a_i / X_k) / (sum x_il a_i / X_l) - 1) / ((sum_{i<=n1_k} t_i a_i / T1_k) / (sum_{i>=n2_l} t_i a_i / T2_l) - 1).

    Parameters
    ----------
    x : units x groups counts
    area : sequence of unit areas

    Returns
    -------
    RichResult
        Keys: DEL, ACO (per group), RCO (groups x groups).

    References
    ----------
    Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.

    Examples
    --------
    >>> round(spatial_concentration([[10, 0], [0, 10]], [1, 3])["DEL"][0], 6)
    0.75
    """
    keep = [i for i, r in enumerate(x) if ssum(float(v) for v in r) > 0]
    X = counts(x)
    a = [float(area[i]) for i in keep]
    n, g = len(X), len(X[0])
    tot, t = col_totals(X), row_totals(X)
    A = ssum(a)
    DEL = [0.5 * ssum(abs(X[i][k] / tot[k] - a[i] / A) for i in range(n)) for k in range(g)]
    order = sorted(range(n), key=lambda i: (a[i], i))
    ts, as_ = [t[i] for i in order], [a[i] for i in order]
    xs = [[X[i][k] for i in order] for k in range(g)]

    def cut_low(k):
        s, i = 0.0, 0
        while s < tot[k]:
            s += ts[i]
            i += 1
        return i, s

    def cut_high(k):
        s, i = 0.0, n
        while s < tot[k]:
            i -= 1
            s += ts[i]
        return i, s

    low = [cut_low(k) for k in range(g)]
    high = [cut_high(k) for k in range(g)]
    v1 = [ssum(xs[k][i] * as_[i] / tot[k] for i in range(n)) for k in range(g)]
    v2 = [ssum(ts[i] * as_[i] / low[k][1] for i in range(low[k][0])) for k in range(g)]
    v3 = [ssum(ts[i] * as_[i] / high[k][1] for i in range(high[k][0], n)) for k in range(g)]
    ACO = [1 - (v1[k] - v2[k]) / (v3[k] - v2[k]) for k in range(g)]
    RCO = [[(v1[p] / v1[q] - 1) / (v2[p] / v3[q] - 1) for q in range(g)] for p in range(g)]
    return RichResult(
        title="Concentration indices", summary_lines=[("groups", g)], payload={"DEL": DEL, "ACO": ACO, "RCO": RCO}
    )


def cheatsheet():
    return "concidx: Massey-Denton concentration DEL, ACO, RCO"
