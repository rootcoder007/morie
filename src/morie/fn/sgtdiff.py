# morie.fn -- k02 batch (rootcoder007/morie)
"""Diffusion (heat) kernel on a graph.

Source consulted: Kondor, R.I. and Lafferty, J. (2002), Diffusion kernels on
graphs and other discrete input spaces, *ICML 2002*, 315-322, equation (5):

    K(beta) = exp(-beta L),    L = D - A

which is the solution at time beta of the heat equation dK/dbeta = -L K with
K(0) = I.  Because L is symmetric positive semi-definite the exponential is
taken through its own eigendecomposition, V diag(exp(-beta lambda)) V', which
is sign-safe (the sign of each eigenvector cancels) and guarantees the kernel
is symmetric positive definite.  L has a zero eigenvalue on each connected
component, so every row of K sums to one -- asserted below.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_diffusion_kernel"]


def sgt_diffusion_kernel(A, beta=1.0):
    """Heat kernel exp(-beta L) of a graph.

    Parameters
    ----------
    A : array-like
        Symmetric adjacency (or weight) matrix.
    beta : float, default 1.0
        Diffusion time.

    Returns
    -------
    RichResult
        estimate (trace of the kernel), kernel, eigenvalues, beta, n, method.
    """
    m = np.atleast_2d(np.asarray(A, dtype=float))
    m = 0.5 * (m + m.T)
    lap = np.diag(np.sum(m, axis=1)) - m
    w, v = np.linalg.eigh(lap)
    k = np.dot(v * np.exp(-float(beta) * w), v.T)
    k = 0.5 * (k + k.T)
    return RichResult(
        payload={
            "estimate": float(np.trace(k)),
            "kernel": k.tolist(),
            "eigenvalues": w.tolist(),
            "beta": float(beta),
            "n": int(m.shape[0]),
            "method": "Graph diffusion (heat) kernel (Kondor & Lafferty 2002, eq. 5)",
        }
    )


# CANONICAL TEST
# >>> A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
# >>> r = sgt_diffusion_kernel(A, 0.7)
# >>> K = r["kernel"]
# >>> assert all(abs(sum(row) - 1.0) < 1e-12 for row in K)   # L 1 = 0
# >>> assert abs(K[0][2] - K[2][0]) < 1e-15                  # symmetric
# >>> # two-node graph has the closed form (1 +/- exp(-2 beta))/2
# >>> r2 = sgt_diffusion_kernel([[0, 1], [1, 0]], 0.3)
# >>> import math
# >>> assert abs(r2["kernel"][0][0] - (1 + math.exp(-0.6)) / 2) < 1e-14


def cheatsheet():
    return "sgtdiff(A, beta): graph diffusion (heat) kernel."


sgtdiffusionkernel = sgt_diffusion_kernel
