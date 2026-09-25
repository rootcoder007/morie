# morie.fn -- function file (rootcoder007/morie)
"""Spectral radius of a spatial weights / adjacency matrix."""

from math import sqrt

from ._richresult import RichResult
from ._spx import dot, matvec, sqmat

__all__ = [
    "spectral_radius",
    "specrad",
    "spectralradius",
]


def spectral_radius(g, iters=400):
    """Largest eigenvalue modulus of a symmetric weights matrix.

    Why a spatial package cares: Schabenberger & Gotway (2005) Sec. 6.2.2.1,
    p. 336, requires (I - rho W) to be non-singular for the SAR model of
    eqs (6.36)-(6.38) to be defined, and states the resulting restriction
    through the eigenvalues of W --

        if theta_min < 0 and theta_max > 0 then
        1/theta_min < rho < 1/theta_max

    (attributed there to Haining 1990, p. 82) -- adding that for a large
    set of identical square regions the extreme eigenvalues approach -4
    and +4, so |rho| < 0.25, and that a row-standardised W has
    theta_max = 1 and theta_min <= -1, so rho < 1 but rho > -1 may be
    violated.

    NOTE: that bound is PROSE in Sec. 6.2.2.1, immediately after eq (6.38).
    It is NOT eq (6.48); eq (6.48) is the Fisher information matrix of the
    CAR model, which merely mentions the eigenvalues of W inside its alpha
    term. Anything in this repository citing (6.48) for the rho interval
    is miscited.

    For a symmetric W the interval is contained in |rho| < 1/rho(W), and
    that bound is returned as ``sar_rho_bound``.

    The computation is power iteration on W^2 from a FIXED, slightly non-uniform
    start vector for a FIXED number of steps -- fixed because an
    early-exit tolerance taken on one language arm and not the other would
    silently change the answer, and non-uniform because an all-ones start
    is orthogonal to the leading eigenvector of some perfectly ordinary
    matrices and fails silently when it is.

    A non-symmetric matrix is REJECTED. Power iteration on one can
    converge to a complex conjugate pair and report a modulus that is not
    the spectral radius; returning a wrong number quietly is worse than
    refusing.

    The generated stub returned the mean of `g` and cited a graph-theory
    monograph. Power iteration is Golub, G. H. & Van Loan, C. F. (2013),
    *Matrix Computations*, 4th edn, Sec. 7.3; the spatial relevance is the
    passage above.

    Parameters
    ----------
    g : (n, n) array-like
        Symmetric adjacency or weights matrix.
    iters : int
        Power-iteration steps.

    Returns
    -------
    RichResult
        ``rho``, ``dominant_eigenvalue``, ``eigenvector``,
        ``sar_rho_bound``, ``n``, ``method``.
    """
    w = sqmat(g, None, "g")
    n = len(w)
    if n < 2:
        raise ValueError("`g` must be at least 2 by 2")
    iters = int(iters)
    if iters < 1:
        raise ValueError("`iters` must be positive")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(w[i][j] - w[j][i]) > 1e-12:
                raise ValueError("`g` must be symmetric; power iteration "
                                 "on a non-symmetric matrix can converge "
                                 "to a complex pair and report a modulus "
                                 "that is not the spectral radius")

    # Iterate on W^2, not W.  A symmetric W can have BOTH +rho and -rho
    # as extreme eigenvalues (every bipartite graph does -- a rook grid,
    # a path), and plain power iteration then oscillates between them
    # and its Rayleigh quotient reports neither.  W^2 has the single
    # dominant eigenvalue rho^2, so ||W v|| converges to rho regardless.
    v = [float((i % 7) + 1) for i in range(n)]
    s = sqrt(dot(v, v))
    v = [t / s for t in v]
    for _ in range(iters):
        u = matvec(w, matvec(w, v))
        s = sqrt(dot(u, u))
        if s < 1e-300:
            raise ValueError("`g` is numerically zero; the spectral "
                             "radius is 0 and no eigenvector is defined")
        v = [t / s for t in u]
    wv = matvec(w, v)
    rho = sqrt(dot(wv, wv))
    if rho <= 0:
        raise ValueError("the spectral radius is 0; `g` has no edges")
    # v lies in the span of the +rho and -rho eigenvectors; split it.
    # A non-negative W always has +rho (Perron-Frobenius); otherwise the
    # component with the larger share names the dominant sign.
    up = [a_ + b_ / rho for a_, b_ in zip(v, wv)]
    um = [a_ - b_ / rho for a_, b_ in zip(v, wv)]
    np_, nm_ = sqrt(dot(up, up)), sqrt(dot(um, um))
    if np_ >= nm_:
        lam, v = rho, [t / np_ for t in up]
    else:
        lam, v = -rho, [t / nm_ for t in um]
    j = 0
    for i in range(n):
        if abs(v[i]) > abs(v[j]):
            j = i
    if v[j] < 0:
        v = [-t for t in v]

    return RichResult(payload={
        "rho": rho,
        "dominant_eigenvalue": lam,
        "eigenvector": v,
        "sar_rho_bound": 1.0 / rho,
        "symmetric": True,
        "iterations": float(iters),
        "n": n,
        "method": ("Spectral radius by power iteration on W^2 (Golub & "
                   "Van Loan 2013, Sec. 7.3); the SAR bound |rho| < 1/rho(W) is "
                   "Schabenberger & Gotway (2005) Sec. 6.2.2.1, p. 336 -- "
                   "NOT eq (6.48)"),
    })


def cheatsheet():
    return "spcrad: spectral radius and the SAR rho bound"


# compact alias per ledger/NAMING.md
specrad = spectral_radius
spectralradius = spectral_radius
