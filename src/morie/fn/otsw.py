# morie.fn -- function file (rootcoder007/morie)
"""Sliced Wasserstein distance between two point clouds."""

from . import _otcore as ot
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ot_sliced_wasserstein"]


def ot_sliced_wasserstein(X, Y, p=2, n_proj=32):
    """Average the one-dimensional distance over many directions.

    Projecting to a line makes the transport problem a sort, so a sliced
    distance costs ``O(L n log n)`` instead of the cubic price of the
    linear program, and it still metrises weak convergence.  The
    directions are a van der Corput / AS 241 sequence rather than a
    pseudo-random one, so the value is reproducible across arms and
    across runs.

    Formula: ``SW_p(mu,nu) = (E_theta [W_p(P_theta mu, P_theta nu)^p])^
    (1/p)`` -- Bonneel et al. (2015) eq. (5); Peyre & Cuturi (2019)
    eq. (10.13), p. 166.

    Parameters
    ----------
    X, Y : array-like, shape (n, d)
        Two point clouds with the same number of points.
    p : float, default 2
        Exponent, positive.
    n_proj : int, default 32
        Number of directions.

    Returns
    -------
    RichResult
        ``SW``, ``SW_p``, ``per_proj`` (the ``W_p^p`` values), ``n``,
        ``d``, ``n_proj``.

    References
    ----------
    Bonneel, N., Rabin, J., Peyre, G. and Pfister, H. (2015).  Sliced and
    Radon Wasserstein barycenters of measures.  Journal of Mathematical
    Imaging and Vision 51(1):22-45.  doi:10.1007/s10851-014-0506-3.
    """
    A = core.mat(X)
    B = core.mat(Y)
    if len(A) != len(B):
        raise ValueError("sliced W_p needs clouds with equal point counts")
    d = len(A[0])
    if len(B[0]) != d:
        raise ValueError("point clouds must share a dimension")
    pp = float(p)
    L = int(n_proj)
    per = []
    for th in ot.directions(d, L):
        px = ot.project(A, th)
        py = ot.project(B, th)
        per.append(ot.wp1d(px, py, pp) ** pp)
    swp = sum(per) / L
    return RichResult(payload={
        "SW": swp ** (1.0 / pp), "SW_p": swp, "per_proj": per,
        "n": len(A), "d": d, "n_proj": L,
        "method": "Sliced Wasserstein distance"})


def cheatsheet():
    return "otsw: sliced Wasserstein distance over deterministic directions"
