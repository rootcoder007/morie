# morie.fn -- k02 batch (rootcoder007/morie)
"""Spectral radius of a graph's adjacency matrix.

Source consulted: Horn, R.A. and Johnson, C.R. (2013), *Matrix Analysis*, 2nd
edition, chapter 8 (Perron-Frobenius).  For a connected graph with a
non-negative symmetric adjacency matrix the largest eigenvalue is simple,
positive, and has a positive eigenvector, and it is bracketed by the mean and
maximum degree -- both bounds are returned so the value can be sanity-checked
without a second eigensolver.  The complete graph K_n has spectral radius
n - 1 exactly, which is the canonical test.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_spectral_radius"]


def sgt_spectral_radius(A):
    """Largest eigenvalue of a symmetric adjacency matrix.

    Parameters
    ----------
    A : array-like
        Symmetric adjacency (or weight) matrix.

    Returns
    -------
    RichResult
        estimate (spectral radius), eigenvalues, perron_vector,
        mean_degree, max_degree, n, method.
    """
    m = np.atleast_2d(np.asarray(A, dtype=float))
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    j = int(np.argmax(np.abs(w)))
    vec = v[:, j]
    s = float(np.sum(vec))
    if s < 0.0:
        vec = -vec
    deg = np.sum(m, axis=1)
    return RichResult(
        payload={
            "estimate": float(np.abs(w[j])),
            "eigenvalues": np.sort(w).tolist(),
            "perron_vector": vec.tolist(),
            "mean_degree": float(np.mean(deg)),
            "max_degree": float(np.max(deg)),
            "n": int(m.shape[0]),
            "method": "Adjacency spectral radius, Perron-Frobenius (Horn & Johnson 2013, ch. 8)",
        }
    )


# CANONICAL TEST
# >>> K4 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
# >>> r = sgt_spectral_radius(K4)
# >>> assert abs(r["estimate"] - 3.0) < 1e-12          # K_n has radius n - 1
# >>> assert r["mean_degree"] <= r["estimate"] <= r["max_degree"] + 1e-12


def cheatsheet():
    return "sgtsbpd(A): adjacency spectral radius."


sgtspectralradius = sgt_spectral_radius
