# morie.fn -- function file (rootcoder007/morie)
"""Multilevel (1-1-1) mediation with within/between decomposition."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["multilevel_mediation"]


def multilevel_mediation(y, x, m, cluster):
    r"""Within- and between-cluster indirect effects.

    For a 1-1-1 design (X, M, Y all measured at level 1) the naive
    pooled indirect effect conflates two distinct quantities. Splitting
    each variable into its cluster mean and its cluster-centred
    deviation,

    .. math:: X_{ij} = \bar X_j + \tilde X_{ij},

    and fitting the mediation paths separately on the two pieces gives
    the within-cluster effect :math:`a_w b_w` and the between-cluster
    effect :math:`a_b b_b`. Zhang, Zyphur and Preacher show the
    uncentred estimate is a *blend* of the two and equals neither
    unless they coincide -- which is why the split matters.

    Parameters
    ----------
    y, x, m : array-like, shape (n,)
        Outcome, treatment, mediator at level 1.
    cluster : array-like, shape (n,)
        Cluster (level 2) identifier.

    Returns
    -------
    RichResult
        keys: ``indirect_within``, ``indirect_between``,
        ``direct_within``, ``direct_between``, ``paths`` dict,
        ``n_clusters``, ``n``, ``method``.

    References
    ----------
    Zhang, Z., Zyphur, M. J. & Preacher, K. J. (2009). Testing
    multilevel mediation using hierarchical linear models: problems
    and solutions. *Organizational Research Methods*, 12(4), 695-719.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    g = np.asarray(cluster).ravel()
    n = y.size
    if not (x.size == n and m.size == n and g.size == n):
        raise ValueError("y, x, m, cluster must have equal length.")
    groups, inv = np.unique(g, return_inverse=True)
    J = groups.size
    if J < 3:
        raise ValueError(f"need at least 3 clusters, got {J}.")

    def split(v):
        means = np.array([v[inv == j].mean() for j in range(J)])
        return means, v - means[inv]

    xb, xw = split(x)
    mb, mw = split(m)
    yb, yw = split(y)

    def ols(cols, t):
        D = np.column_stack([np.ones(t.size), *cols])
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b

    aw = float(ols([xw], mw)[1])
    byw = ols([xw, mw], yw)
    cw, bw = float(byw[1]), float(byw[2])

    ab = float(ols([xb], mb)[1])
    byb = ols([xb, mb], yb)
    cb, bb = float(byb[1]), float(byb[2])

    return RichResult(
        payload={
            "indirect_within": aw * bw,
            "indirect_between": ab * bb,
            "direct_within": cw,
            "direct_between": cb,
            "paths": {"a_within": aw, "b_within": bw, "a_between": ab, "b_between": bb},
            "n_clusters": int(J),
            "n": int(n),
            "method": "1-1-1 multilevel mediation (within/between decomposition)",
        }
    )


def cheatsheet():
    return "mlmMd: split into cluster means + deviations; a_w b_w and a_b b_b"
