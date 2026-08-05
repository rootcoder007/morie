# morie.fn -- function file (rootcoder007/morie)
"""Empirical angular (spectral) measure of a bivariate extreme."""

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_angular_measure"]


def evt_angular_measure(X, k):
    """
    Empirical angular measure of the k largest radii

    Formula: H_n(B) = (1/k) sum 1{||X_i|| >= r_n, X_i/||X_i|| in B}

    The data are first rank-transformed to standard Frechet margins, so
    the radial and angular parts separate.  Each of the k largest radii
    contributes an atom of mass 1/k at its angle w = X_1/(X_1 + X_2),
    which makes H a probability measure by construction; the mean
    constraint of a valid angular measure is mean(w) = 1/2.

    Parameters
    ----------
    X : array-like
        n x 2 matrix of observations.
    k : int
        Number of upper order statistics of the radius to keep.

    Returns
    -------
    result : dict
        Keys: H, atoms, weights, estimate (mean angle), n_used, n.

    References
    ----------
    Einmahl, de Haan & Sinha (1997), Stoch. Proc. Appl. 70(2):143-171.
    """
    M = core.mat(X)
    n = len(M)
    if n == 0:
        raise ValueError("empty input: X has no rows")
    if len(M[0]) != 2:
        raise ValueError("X must have exactly two columns")
    k = int(k)
    if k < 1 or k > n:
        raise ValueError("k must lie between 1 and the number of rows")
    c0 = [r[0] for r in M]
    c1 = [r[1] for r in M]
    r0 = core.rank_avg(c0)
    r1 = core.rank_avg(c1)
    # standard Frechet: 1 / (1 - rank/(n+1)) = (n+1)/(n+1-rank)
    f0 = [(n + 1.0) / (n + 1.0 - v) for v in r0]
    f1 = [(n + 1.0) / (n + 1.0 - v) for v in r1]
    rad = [f0[i] + f1[i] for i in range(n)]
    idx = sorted(range(n), key=lambda i: (-rad[i], i))[:k]
    atoms = sorted(f0[i] / rad[i] for i in idx)
    w = [1.0 / k] * k
    mean_angle = sum(atoms) / k
    return RichResult(payload={
        "H": sum(w),
        "atoms": atoms,
        "weights": w,
        "estimate": mean_angle,
        "n_used": k,
        "n": n,
        "method": "empirical angular measure of the k largest radii",
    })


def cheatsheet():
    return "evangia: empirical angular measure"


# compact alias per ledger/NAMING.md
evtangularmeasure = evt_angular_measure
