# morie.fn -- k02 batch (rootcoder007/morie)
"""Katz centrality.

Source consulted: Katz, L. (1953), A new status index derived from sociometric
analysis, *Psychometrika* 18(1), 39-43.  Katz counts every walk into a node,
discounting a walk of length k by alpha^k:

    x = sum_{k>=1} alpha^k A^k 1 = ( (I - alpha A)^-1 - I ) 1

which converges when alpha < 1 / rho(A).  ``alpha`` defaults to half the
convergence radius.  The series form and the linear-solve form are computed
independently and their agreement is asserted in the canonical test, so the
inverse is checked against the definition rather than trusted.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_katz_centrality"]


def sgt_katz_centrality(A, alpha=None, beta=1.0):
    """Katz status index.

    Parameters
    ----------
    A : array-like
        Adjacency matrix.
    alpha : float, optional
        Attenuation factor; half of 1/rho(A) if omitted.
    beta : float, default 1.0
        Exogenous status added to every node.

    Returns
    -------
    RichResult
        estimate (largest centrality), centrality, normalized, alpha,
        radius, n, method.
    """
    m = np.atleast_2d(np.asarray(A, dtype=float))
    n = m.shape[0]
    sym = 0.5 * (m + m.T)
    rho = float(np.max(np.abs(np.linalg.eigvalsh(sym))))
    a = 0.5 / rho if alpha is None else float(alpha)
    if rho > 0.0 and a * rho >= 1.0:
        raise ValueError("alpha must be below 1 / spectral radius")
    one = np.ones(n) * float(beta)
    x = np.linalg.solve(np.eye(n) - a * m, one) - one
    tot = float(np.sum(np.abs(x)))
    return RichResult(
        payload={
            "estimate": float(np.max(x)),
            "centrality": x.tolist(),
            "normalized": (x / tot).tolist() if tot > 0.0 else x.tolist(),
            "alpha": float(a),
            "radius": rho,
            "n": int(n),
            "method": "Katz centrality (Katz 1953)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
# >>> r = sgt_katz_centrality(A, alpha=0.1)
# >>> # the truncated walk series must agree with the linear solve
# >>> import functools
# >>> M = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
# >>> acc = [0.0, 0.0, 0.0]
# >>> pw = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
# >>> for k in range(1, 60):
# ...     pw = [[sum(pw[i][t] * M[t][j] for t in range(3)) for j in range(3)] for i in range(3)]
# ...     acc = [acc[i] + 0.1 ** k * sum(pw[i]) for i in range(3)]
# >>> assert max(abs(acc[i] - r["centrality"][i]) for i in range(3)) < 1e-12


def cheatsheet():
    return "sgtkem(A, alpha): Katz centrality."


sgtkatzcentrality = sgt_katz_centrality
