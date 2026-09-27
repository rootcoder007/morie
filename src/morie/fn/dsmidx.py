"""Dissimilarity and segregation indices (Duncan and Duncan 1955; Morrill 1991).

Duncan, O. D. and Duncan, B. (1955). A methodological analysis of segregation indexes.
American Sociological Review 20, 210-217. Morrill, R. L. (1991). On the measure of
geographic segregation. Geography Research Forum 11, 25-36. Massey, D. S. and Denton, N. A.
(1988). The dimensions of residential segregation. Social Forces 67, 281-315.
"""

from ._richresult import RichResult
from ._segcore import col_totals, counts, ssum

__all__ = ["dissimilarity_index"]


def dissimilarity_index(x, contiguity=None):
    r"""Evenness: D_kl = 1/2 sum_i |x_ik / X_k - x_il / X_l| between every pair of groups, and the
    segregation index IS_k (group k against everyone else).

    With a contiguity matrix c (0/1, zero diagonal), Morrill's adjusted index subtracts the
    average absolute difference in the group-k share z_i = x_ik / (x_ik + x_il) across
    adjacent units: D(adj)_kl = D_kl - sum_ij c_ij |z_i - z_j| / sum_ij c_ij.

    Parameters
    ----------
    x : units x groups counts
    contiguity : n x n matrix, optional

    Returns
    -------
    RichResult
        Keys: D (groups x groups), IS (per group), D_morrill (when ``contiguity`` is given).

    References
    ----------
    Duncan, O. D. and Duncan, B. (1955). American Sociological Review 20, 210-217.
    Morrill, R. L. (1991). Geography Research Forum 11, 25-36.

    Examples
    --------
    >>> dissimilarity_index([[10, 0], [0, 10]])["D"][0][1]
    1.0
    """
    X = counts(x)
    n, g = len(X), len(X[0])
    tot = col_totals(X)
    D = [[0.0] * g for _ in range(g)]
    for a in range(g):
        for b in range(a + 1, g):
            D[a][b] = D[b][a] = 0.5 * ssum(abs(r[a] / tot[a] - r[b] / tot[b]) for r in X)
    rows = [ssum(r) for r in X]
    IS = []
    for k in range(g):
        rest = ssum(rows) - tot[k]
        IS.append(0.5 * ssum(abs(X[i][k] / tot[k] - (rows[i] - X[i][k]) / rest) for i in range(n)))
    out = {"D": D, "IS": IS}
    if contiguity is not None:
        c = [[float(v) for v in r] for r in contiguity]
        if len(c) != len(x):
            raise ValueError("contiguity must be n x n over the input units")
        keep = [i for i, r in enumerate(x) if ssum(float(v) for v in r) > 0]
        c = [[c[i][j] for j in keep] for i in keep]
        csum = ssum(ssum(r) for r in c)
        Dm = [[0.0] * g for _ in range(g)]
        for a in range(g):
            for b in range(a + 1, g):
                z = [X[i][a] / (X[i][a] + X[i][b]) if X[i][a] + X[i][b] > 0 else 0.0 for i in range(n)]
                adj = ssum(c[i][j] * abs(z[i] - z[j]) for i in range(n) for j in range(n))
                Dm[a][b] = Dm[b][a] = D[a][b] - adj / csum
        out["D_morrill"] = Dm
    return RichResult(title="Dissimilarity index", summary_lines=[("groups", g)], payload=out)


def cheatsheet():
    return "dsmidx: Duncan dissimilarity, one-vs-rest segregation and Morrill's contiguity-adjusted D"
