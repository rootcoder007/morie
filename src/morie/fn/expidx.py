"""Exposure (interaction) indices (Bell 1954; Lieberson 1981; Morgan 1983).

Massey, D. S. and Denton, N. A. (1988). The dimensions of residential segregation. Social
Forces 67, 281-315. Morgan, B. S. (1983). An alternate approach to the development of a
distance-based measure of racial segregation. American Journal of Sociology 88, 1237-1249.
"""

from ._richresult import RichResult
from ._segcore import col_totals, counts, decay, row_totals, ssum

__all__ = ["exposure_index"]


def exposure_index(x, exact=False, distance=None, beta=1.0):
    r"""Interaction xPy_kl = sum_i (x_ik / X_k)(x_il / t_i): the chance that a member of group k
    meets a member of group l in their unit (``exact=True`` uses (x_il - [k = l]) / (t_i - 1),
    sampling without replacement).

    With a distance matrix d, Morgan's distance-decay interaction replaces the unit share
    by sum_j K_ij x_jl / t_j with K_ij = t_j exp(-beta d_ij) / sum_j t_j exp(-beta d_ij).

    Parameters
    ----------
    x : units x groups counts
    exact : bool
    distance : n x n matrix, optional
    beta : float

    Returns
    -------
    RichResult
        Keys: xPy (groups x groups; the diagonal is isolation), DPxy (with ``distance``).

    References
    ----------
    Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.
    Morgan, B. S. (1983). American Journal of Sociology 88, 1237-1249.

    Examples
    --------
    >>> exposure_index([[5, 5], [5, 5]])["xPy"][0][1]
    0.5
    """
    X = counts(x)
    n, g = len(X), len(X[0])
    tot, t = col_totals(X), row_totals(X)
    P = [[0.0] * g for _ in range(g)]
    for a in range(g):
        for b in range(g):
            if exact:
                P[a][b] = ssum(X[i][a] / tot[a] * (X[i][b] - (1.0 if a == b else 0.0)) / (t[i] - 1) for i in range(n))
            else:
                P[a][b] = ssum(X[i][a] / tot[a] * X[i][b] / t[i] for i in range(n))
    out = {"xPy": P}
    if distance is not None:
        E = decay(distance, float(beta))
        K1 = [ssum(E[i][j] * t[j] for j in range(n)) for i in range(n)]
        K = [[E[i][j] * t[j] / K1[i] for j in range(n)] for i in range(n)]
        DP = [[0.0] * g for _ in range(g)]
        for a in range(g):
            for b in range(g):
                share = [ssum(K[i][j] * X[j][b] / t[j] for j in range(n)) for i in range(n)]
                DP[a][b] = ssum(X[i][a] / tot[a] * share[i] for i in range(n))
        out["DPxy"] = DP
    return RichResult(title="Exposure index", summary_lines=[("groups", g)], payload=out)


def cheatsheet():
    return "expidx: interaction xPy and Morgan distance-decay exposure"
