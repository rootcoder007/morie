# morie.fn -- k02 batch (rootcoder007/morie)
"""HITS: Kleinberg's hub and authority scores.

Source consulted: Kleinberg, J.M. (1999), Authoritative sources in a
hyperlinked environment, *Journal of the ACM* 46(5), 604-632, section 3.  The
mutually reinforcing relation a = A' h, h = A a makes the authority vector the
principal eigenvector of A'A and the hub vector that of A A'.  Computed by
power iteration from the uniform start with a fixed number of steps, and
scaled to a maximum of one, which is the normalisation ``igraph``'s
``hits_scores`` reports.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_hits_kleinberg"]


def sgt_hits_kleinberg(A, iters=200):
    """Hub and authority scores.

    Parameters
    ----------
    A : array-like
        Adjacency matrix; entry (i, j) is a link from i to j.
    iters : int, default 200
        Power-iteration steps (fixed, so the result is deterministic).

    Returns
    -------
    RichResult
        estimate (largest authority), authority, hub, eigenvalue, n, method.
    """
    m = np.atleast_2d(np.asarray(A, dtype=float))
    n = m.shape[0]
    ata = np.dot(m.T, m)
    aat = np.dot(m, m.T)
    a = np.ones(n)
    h = np.ones(n)
    lam = 0.0
    for _ in range(int(iters)):
        a2 = np.dot(ata, a)
        nm = float(np.sqrt(np.sum(a2 * a2)))
        if nm == 0.0:
            break
        a = a2 / nm
        h2 = np.dot(aat, h)
        nh = float(np.sqrt(np.sum(h2 * h2)))
        if nh > 0.0:
            h = h2 / nh
        lam = nm
    amax = float(np.max(np.abs(a)))
    hmax = float(np.max(np.abs(h)))
    if float(np.sum(a)) < 0.0:
        a = -a
    if float(np.sum(h)) < 0.0:
        h = -h
    return RichResult(
        payload={
            "estimate": 1.0 if amax > 0.0 else 0.0,
            "authority": (a / amax).tolist() if amax > 0.0 else a.tolist(),
            "hub": (h / hmax).tolist() if hmax > 0.0 else h.tolist(),
            "eigenvalue": float(lam),
            "n": int(n),
            "method": "HITS hub and authority scores (Kleinberg 1999, sec. 3)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
# ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
# >>> r = sgt_hits_kleinberg(A)
# >>> # igraph hits_scores on the same graph
# >>> assert abs(r["authority"][0] - 0.707106781186547) < 1e-9
# >>> assert abs(r["authority"][2] - 1.0) < 1e-12


def cheatsheet():
    return "sgthits(A): HITS hub and authority scores."


sgthitskleinberg = sgt_hits_kleinberg
