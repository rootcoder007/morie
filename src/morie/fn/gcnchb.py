# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""ChebNet: spectral filtering by a truncated Chebyshev expansion.

Defferrard, Bresson and Vandergheynst (2016), "Convolutional neural
networks on graphs with fast localized spectral filtering", NIPS 29,
arXiv:1606.09375, equations (4) and (5):

    g_theta(L) x = sum_{k=0}^{K-1} theta_k T_k(Lt) x,
    Lt = 2 L / lambda_max - I_n,
    T_k(x) = 2 x T_{k-1}(x) - T_{k-2}(x),  T_0 = 1, T_1 = x.

The recurrence is what makes the filter K-localised and costs only K
sparse products.  lambda_max is taken from the Jacobi eigenvalues of
the supplied (symmetric) Laplacian.  With no coefficients passed the
deterministic choice theta_k = 1/(k+1) is used.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["chebnet"]


def chebnet(L, X, K=3, theta=None):
    """Chebyshev polynomial filter of order K-1 applied to X."""
    M = core.mat(L)
    n = len(M)
    if n == 0:
        raise ValueError("chebnet: Laplacian is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("chebnet: Laplacian must be square")
    H = core.mat(X)
    if len(H) != n:
        raise ValueError("chebnet: X must have one row per node")
    order = int(K)
    if order < 1:
        raise ValueError("chebnet: K must be at least 1")
    th = [1.0 / (kk + 1.0) for kk in range(order)] if theta is None else core.vec(theta)
    if len(th) != order:
        raise ValueError("chebnet: theta must have K entries")
    vals, _ = core.jacobi(M)
    lmax = max(vals)
    if lmax <= 0:
        raise ValueError("chebnet: Laplacian has no positive eigenvalue")
    Lt = [[2.0 * M[i][j] / lmax - (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    p = len(H[0])
    Tprev = [row[:] for row in H]
    out = [[th[0] * Tprev[i][j] for j in range(p)] for i in range(n)]
    if order > 1:
        Tcur = core.matmul(Lt, H)
        for i in range(n):
            for j in range(p):
                out[i][j] += th[1] * Tcur[i][j]
        for kk in range(2, order):
            Tn = core.matmul(Lt, Tcur)
            Tn = [[2.0 * Tn[i][j] - Tprev[i][j] for j in range(p)] for i in range(n)]
            for i in range(n):
                for j in range(p):
                    out[i][j] += th[kk] * Tn[i][j]
            Tprev, Tcur = Tcur, Tn
    flat = [v for row in out for v in row]
    return RichResult(
        title="ChebNet spectral filter",
        summary_lines=[("nodes", n), ("K", order), ("lambda_max", lmax)],
        payload={
            "estimate": sum(flat) / len(flat),
            "H": out,
            "lambda_max": lmax,
            "K": order,
            "n": n,
            "method": "sum_k theta_k T_k(2L/lmax - I) X, Defferrard et al. (2016) eqs. (4)-(5)",
        },
    )
