# morie.fn -- function file (rootcoder007/morie)
"""Sliced Wasserstein distance from quantile functions."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_sliced_distance_quant"]


def ot_sliced_distance_quant(X, Y, p=2, n_proj=32):
    """Sliced distance computed on a shared quantile grid.

    Rabin et al. build their texture-mixing barycenter out of projected
    quantile functions, which is what lets the two clouds have different
    sizes: instead of pairing order statistics one for one, both
    projections are evaluated at a common grid of probabilities.  When the
    clouds do have the same size the grid reproduces the order statistics
    exactly, so this agrees with the sorted estimator.

    Formula: ``SW_p^p = E_theta int_0^1 |F^-1_{theta,mu}(q) -
    F^-1_{theta,nu}(q)|^p dq``, the integral taken on the midpoint grid
    ``q_k = (k - 1/2)/G`` -- Rabin et al. (2012) Section 3.

    Parameters
    ----------
    X, Y : array-like, shape (n, d), (m, d)
        Point clouds; the counts may differ.
    p : float, default 2
        Exponent, positive.
    n_proj : int, default 32
        Number of directions.

    Returns
    -------
    RichResult
        ``SW``, ``SW_p``, ``per_proj``, ``n``, ``m``, ``d``, ``n_proj``,
        ``grid_size``.

    References
    ----------
    Rabin, J., Peyre, G., Delon, J. and Bernot, M. (2012).  Wasserstein
    barycenter and its application to texture mixing.  Lecture Notes in
    Computer Science 6667:435-446.  doi:10.1007/978-3-642-24785-9_37.
    """
    A = core.mat(X)
    B = core.mat(Y)
    d = len(A[0])
    if len(B[0]) != d:
        raise ValueError("point clouds must share a dimension")
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        raise ValueError("empty point cloud")
    pp = float(p)
    if pp <= 0.0:
        raise ValueError("p must be positive")
    L = int(n_proj)
    G = n if n > m else m
    grid = [(k + 0.5) / G for k in range(G)]
    per = []
    for th in ot.directions(d, L):
        qx = ot.quantiles(ot.project(A, th), grid)
        qy = ot.quantiles(ot.project(B, th), grid)
        per.append(sum(abs(qx[k] - qy[k]) ** pp for k in range(G)) / G)
    swp = sum(per) / L
    return RichResult(payload={
        "SW": swp ** (1.0 / pp), "SW_p": swp, "per_proj": per,
        "n": n, "m": m, "d": d, "n_proj": L, "grid_size": G,
        "method": "Quantile-based sliced Wasserstein distance"})


def cheatsheet():
    return "otsd: sliced Wasserstein distance from projected quantile functions"
