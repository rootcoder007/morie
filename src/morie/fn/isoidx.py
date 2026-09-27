"""Isolation index and correlation ratio (Bell 1954; White 1986; Morgan 1983).

Massey, D. S. and Denton, N. A. (1988). The dimensions of residential segregation. Social
Forces 67, 281-315. White, M. J. (1986). Segregation and diversity measures in population
distribution. Population Index 52, 198-221.
"""

from ._richresult import RichResult
from ._segcore import col_totals, counts, ssum
from .expidx import exposure_index

__all__ = ["isolation_index"]


def isolation_index(x, exact=False, distance=None, beta=1.0):
    r"""Isolation xPx_k = sum_i (x_ik / X_k)(x_ik / t_i) and the correlation ratio
    eta2_k = (xPx_k - P_k) / (1 - P_k), P_k the group's overall share (White 1986);
    with ``distance`` also Morgan's distance-decay isolation DPxx.

    Parameters
    ----------
    x : units x groups counts
    exact : bool
    distance : n x n matrix, optional
    beta : float

    Returns
    -------
    RichResult
        Keys: xPx, eta2, DPxx (with ``distance``).

    References
    ----------
    Massey, D. S. and Denton, N. A. (1988). Social Forces 67, 281-315.
    White, M. J. (1986). Population Index 52, 198-221.

    Examples
    --------
    >>> isolation_index([[10, 0], [0, 10]])["xPx"]
    [1.0, 1.0]
    """
    X = counts(x)
    g = len(X[0])
    e = exposure_index(x, exact=exact, distance=distance, beta=beta)
    tot = col_totals(X)
    N = ssum(tot)
    xpx = [e["xPy"][k][k] for k in range(g)]
    out = {"xPx": xpx, "eta2": [(xpx[k] - tot[k] / N) / (1 - tot[k] / N) for k in range(g)]}
    if distance is not None:
        out["DPxx"] = [e["DPxy"][k][k] for k in range(g)]
    return RichResult(title="Isolation index", summary_lines=[("groups", g)], payload=out)


def cheatsheet():
    return "isoidx: isolation xPx, correlation ratio eta2, Morgan distance-decay isolation"
