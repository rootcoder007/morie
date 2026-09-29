# morie.fn -- function file (rootcoder007/morie)
"""Fredholm integral equation of the first kind (statistical inverse problem)."""

from ._qpcore import inverse, ssum
from ._richresult import RichResult


def horowitz_fredholm_eq(m, k, alpha=1e-3, weights=None):
    r"""Tikhonov-regularised solution of a discretised Fredholm equation of the first kind.

    ``m(w) = int k(x, w) g(x) dx`` is discretised on grids ``w_i`` and
    ``x_j`` with quadrature weights ``d_j`` (default 1): ``m = K D g``. The
    problem is ill-posed -- ``K``'s singular values decay, so ``(KD)^{-1}``
    amplifies noise -- and the Tikhonov estimate ``g = (A'A + alpha I)^{-1}
    A'm``, ``A = K D``, trades bias for stability (Tikhonov 1963; Horowitz
    2009, ch. 5, eq. 5.1 and sec. 5.2). Returns ``g_hat`` and the residual
    norm ``||A g - m||``.

    Parameters
    ----------
    m : array-like, shape (n_w,)
        Left-hand side on the ``w`` grid.
    k : array-like, shape (n_w, n_x)
        Kernel values ``k(x_j, w_i)``.
    alpha : float
        Tikhonov regularisation parameter (> 0).
    weights : array-like, shape (n_x,), optional
        Quadrature weights of the ``x`` grid.

    References
    ----------
    Tikhonov, A. N. (1963). Solution of incorrectly formulated problems and
    the regularization method. *Soviet Mathematics Doklady* 4, 1035-1038.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*, ch. 5. Springer.

    Examples
    --------
    >>> K = [[1.0, 0.5], [0.5, 1.0], [0.2, 0.9]]
    >>> r = horowitz_fredholm_eq([1.5, 1.5, 1.1], K, alpha=1e-12)
    >>> [round(v, 8) for v in r["g_hat"]]
    [1.0, 1.0]
    """
    mv = [float(v) for v in (m.tolist() if hasattr(m, "tolist") else m)]
    K = [[float(v) for v in r] for r in (k.tolist() if hasattr(k, "tolist") else k)]
    nw, nx = len(K), len(K[0])
    if len(mv) != nw:
        raise ValueError("m must have one value per row of k")
    if not alpha > 0:
        raise ValueError("alpha must be positive")
    d = [1.0] * nx if weights is None else [float(v) for v in weights]
    A = [[K[i][j] * d[j] for j in range(nx)] for i in range(nw)]
    G = inverse(
        [
            [ssum(A[i][p] * A[i][q] for i in range(nw)) + (alpha if p == q else 0.0) for q in range(nx)]
            for p in range(nx)
        ]
    )
    rhs = [ssum(A[i][p] * mv[i] for i in range(nw)) for p in range(nx)]
    g = [ssum(G[p][q] * rhs[q] for q in range(nx)) for p in range(nx)]
    res = [ssum(A[i][j] * g[j] for j in range(nx)) - mv[i] for i in range(nw)]
    return RichResult(
        payload={
            "g_hat": g,
            "residual_norm": ssum(v * v for v in res) ** 0.5,
            "alpha": alpha,
            "method": "Tikhonov-regularised Fredholm equation of the first kind",
        }
    )


def cheatsheet():
    return "hrzfrd: Tikhonov solution g = (A'A + alpha I)^-1 A'm of m = K D g (Fredholm, first kind)"
