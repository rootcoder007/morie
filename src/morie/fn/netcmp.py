# morie.fn -- slice s03 (rootcoder007/morie)
"""Graph comparison by a choice of kernel.

Sources consulted: Shervashidze, N. et al. (2011).  Weisfeiler-Lehman
graph kernels.  *JMLR* 12, 2539-2561 (FETCHED) for the WL subtree kernel
and for the survey of the alternatives; Gaertner, T., Flach, P. and
Wrobel, S. (2003).  On graph kernels: hardness results and efficient
alternatives.  *COLT/Kernel* 2777, 129-143, for the geometric random-walk
kernel

    k_RW(G, G') = sum_(i,j) [ sum_(l=0)^inf lambda^l A_x^l ]_(ij)
                = sum_(i,j) [ (I - lambda A_x)^(-1) ]_(ij)

on the direct product graph, which converges for lambda below the
reciprocal of the largest eigenvalue of A_x.  The 2003 COLT paper is
paywalled; the closed form is quoted in its standard published form and
is restated in section 2 of Shervashidze et al.

All three kernels are normalised to a cosine, k / sqrt(k11 k22), so the
value is comparable across graph sizes -- which is the point of a
comparison.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .weisL import wl_kernel
from .grafl import graphlet_kernel

__all__ = ["network_comparison"]


def _rw(G1, G2, lam=0.1):
    A = k.mat(G1)
    B = k.mat(G2)
    n = len(A)
    m = len(B)
    N = n * m
    M = [[0.0] * N for _ in range(N)]
    for i in range(n):
        for j in range(m):
            for a in range(n):
                for b in range(m):
                    M[i * m + j][a * m + b] = -lam * A[i][a] * B[j][b]
    for i in range(N):
        M[i][i] += 1.0
    ones = [1.0] * N
    x = k.ridgesolve(M, ones, 1e-12) if N else []
    s = 0.0
    for v in x:
        s += v
    return s


def network_comparison(G1, G2, kernel="wl", h=3, k_size=3, lam=0.1):
    """Normalised kernel similarity between two graphs.

    Parameters
    ----------
    G1, G2 : 2-D array-like
        Adjacency matrices.
    kernel : {"wl", "graphlet", "rw"}
        Which kernel to use.
    h : int
        WL iterations.
    k_size : int
        Graphlet size.
    lam : float
        Random-walk decay.

    Returns
    -------
    RichResult with payload:
        estimate : the cosine-normalised kernel
        raw      : the unnormalised kernel
        k11, k22 : the two self-similarities
    """
    if kernel == "graphlet":
        f = lambda a, b: graphlet_kernel(a, b, k_size, True)["estimate"]
    elif kernel == "rw":
        f = lambda a, b: _rw(a, b, lam)
    else:
        f = lambda a, b: wl_kernel(a, b, h)["estimate"]
    raw = f(G1, G2)
    k11 = f(G1, G1)
    k22 = f(G2, G2)
    d = math.sqrt(k11 * k22) if k11 > 0.0 and k22 > 0.0 else 0.0
    return RichResult(
        title="Graph kernel comparison",
        summary_lines=[("kernel", kernel), ("cosine", raw / d if d > 0.0 else float("nan"))],
        payload={
            "estimate": raw / d if d > 0.0 else float("nan"),
            "raw": raw,
            "k11": k11,
            "k22": k22,
            "kernel": kernel,
            "method": "Cosine-normalised graph kernel (WL, graphlet, or geometric random walk)",
        },
    )


def cheatsheet():
    return "netcmp: Graph kernel comparison"
