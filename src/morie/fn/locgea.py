# morie.fn -- function file (rootcoder007/morie)
"""Local Geary's C, univariate and multivariate (Anselin 1995, 2019)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult

__all__ = ["local_geary"]


def _standardise(col):
    n = len(col)
    m = sum(col) / n
    sd = (sum((v - m) ** 2 for v in col) / (n - 1)) ** 0.5
    if sd == 0.0:
        raise ValueError("a variable is constant")
    return [(v - m) / sd for v in col]


def local_geary(x, W) -> SpatialResult:
    r"""Local Geary statistic of each unit.

    With every variable standardised (mean 0, sample standard deviation 1),

    .. math::

        C_i = \frac{1}{k} \sum_{v=1}^{k} \sum_j w_{ij} (z_{iv} - z_{jv})^2 ,

    the univariate form of Anselin (1995) for ``k = 1`` and the
    multivariate form of Anselin (2019) otherwise, as ``spdep::localC``
    (a vector, or a list of variables).  Small values mark a unit similar
    to its neighbours, large values a dissimilar one.

    :param x: Values (n,), or (n, k) with one variable per column.
    :param W: Spatial weights (n, n); the diagonal is ignored.
    :return: :class:`SpatialResult` with ``local_values`` the (n,) ``C_i``
        and ``statistic`` their sum.

    References
    ----------
    Anselin, L. (1995). Local indicators of spatial association -- LISA.
    *Geographical Analysis*, 27(2), 93-115.
    Anselin, L. (2019). A local indicator of multivariate spatial
    association: extending Geary's c. *Geographical Analysis*, 51(2),
    133-150.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> [round(c, 6) for c in local_geary([1.0, 2.0, 4.0, 8.0], W).local_values]
    [0.104348, 0.521739, 2.086957, 1.669565]
    """
    xa = np.asarray(x, dtype=float)
    cols = [xa.tolist()] if xa.ndim == 1 else [list(c) for c in zip(*xa.tolist())]
    n = len(cols[0])
    Wl = np.asarray(W, dtype=float).tolist()
    if len(Wl) != n or any(len(r) != n for r in Wl):
        raise ValueError("W must be n x n with n the number of units")
    if n < 2:
        raise ValueError("need at least two units")
    zs = [_standardise(c) for c in cols]
    k = len(zs)
    out = []
    for i in range(n):
        s = 0.0
        for z in zs:
            zi = z[i]
            for j in range(n):
                if j != i:
                    s += Wl[i][j] * (zi - z[j]) ** 2
        out.append(s / k)
    return SpatialResult(
        name="local_geary",
        statistic=sum(out),
        local_values=out,
        extra={"n_variables": k},
    )


def cheatsheet() -> str:
    return "local_geary(x, W) -> local Geary C_i, univariate or multivariate (spdep::localC)."
