"""Multivariate outlier rule based on a robust center and scatter (MVE by default).

Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (13.2).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qchisq

__all__ = ["outmah"]


def outmah(X, center=None, scatter=None, q=0.975):
    """Flag X_i when sqrt((X_i - C) M^{-1} (X_i - C)') > sqrt(chi^2_{q,p}) (13.2).

    C and M default to the minimum volume ellipsoid estimates (morie's ``mve``);
    any robust center and scatter can be supplied instead.

    Parameters
    ----------
    X : n x p nested sequence
    center : sequence of p floats, optional
    scatter : p x p nested sequence, optional
    q : float
        Chi-squared quantile.

    Returns
    -------
    RichResult
        Keys: outliers (0-based indices), distance, crit.

    References
    ----------
    Rousseeuw, P. J. & van Zomeren, B. C. (1990). JASA 85, 633-639.
    Wilcox, R. R. (2017). Modern Statistics for the Social and Behavioral
    Sciences (2nd ed.). CRC Press. Eq (13.2).

    Examples
    --------
    >>> outmah([[0, 0], [1, 0], [0, 1], [9, 9]], center=[0, 0], scatter=[[1, 0], [0, 1]])["outliers"]
    [3]
    """
    X = [[float(v) for v in row] for row in X]
    p = len(X[0])
    if center is None or scatter is None:
        from .mvedet import mve

        est = mve(X)
        center = list(est["center"]) if center is None else center
        scatter = est["cov"] if scatter is None else scatter
        scatter = scatter.tolist() if hasattr(scatter, "tolist") else scatter
    c = [float(v) for v in center]
    M = [[float(v) for v in row] for row in scatter]
    # invert M by Gauss-Jordan with partial pivoting
    A = [M[i][:] + [float(i == j) for j in range(p)] for i in range(p)]
    for k in range(p):
        piv = max(range(k, p), key=lambda i: abs(A[i][k]))
        if abs(A[piv][k]) < 1e-300:
            raise ValueError("scatter matrix is singular")
        A[k], A[piv] = A[piv], A[k]
        f = A[k][k]
        A[k] = [v / f for v in A[k]]
        for i in range(p):
            if i != k and A[i][k] != 0:
                g = A[i][k]
                A[i] = [a - g * b for a, b in zip(A[i], A[k])]
    inv = [row[p:] for row in A]
    dist = []
    for row in X:
        e = [a - b for a, b in zip(row, c)]
        dist.append(math.sqrt(max(0.0, sum(e[i] * inv[i][j] * e[j] for i in range(p) for j in range(p)))))
    crit = math.sqrt(qchisq(q, p))
    return RichResult(
        title="Outliers by robust Mahalanobis distance",
        summary_lines=[("outliers", sum(d > crit for d in dist)), ("crit", crit)],
        payload={"outliers": [i for i, d in enumerate(dist) if d > crit], "distance": dist, "crit": crit},
    )


def cheatsheet():
    return "outmah: flag robust Mahalanobis distances above sqrt(chi^2_.975,p), MVE default. Wilcox (2017) eq (13.2)."
