# morie.fn -- function file (rootcoder007/morie)
"""Second-order (residual) bias correction on top of a first-order TMLE."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_residual"]


def tmle_residual(y, D, X):
    """First-order TMLE plus an estimated second-order remainder.

    A first-order TMLE is unbiased to first order and leaves a
    second-order remainder that behaves like the PRODUCT of the two
    nuisance errors: it vanishes if either model is right, but when both
    are only approximately right at slow rates it does not vanish fast
    enough for root-n inference.  The residual (second-order) estimator
    estimates that remainder directly with a U-statistic over pairs,
    using a finite-dimensional projection kernel

        ``K_k(x, x') = phi(x)' Omega^{-1} phi(x')``,
        ``Omega = n^{-1} sum_i phi(X_i) phi(X_i)'``,

    and adds it back:

        ``IF22 = -[n(n-1)]^{-1} sum_{i != j} a_i K_k(X_i, X_j) b_j``,
        ``a_i = (D_i - g_i) / [g_i (1 - g_i)]``,
        ``b_j = H_j (y_j - Q*_j)``.

    The basis is fixed at ``[1, X, X^2]`` elementwise -- ``k`` is the
    tuning parameter of the whole method, and a larger basis buys bias
    reduction at the cost of variance.  When the outcome regression fits
    exactly the residuals are zero, ``b`` is zero, and the correction is
    identically zero: that degenerate case is the cheapest way to check
    the sign convention.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.

    Returns
    -------
    RichResult
        ``estimate`` (corrected), ``se``, ``psi1`` (first order),
        ``if22`` (the correction), ``k_basis``, ``n``.

    References
    ----------
    Robins, J., Li, L., Mukherjee, R., Tchetgen Tchetgen, E. & van der
    Vaart, A. (2017).  Minimax estimation of a functional on a
    structured high-dimensional model.  Annals of Statistics 45(5).
    doi:10.1214/16-AOS1515.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    if n < 3 or len(Dv) != n:
        raise ValueError("tmle_residual: y and D must share one length >= 3")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_residual: X must have one row per subject")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    base = S.tmle(yv, Dv, W)
    g = base["g"]
    H = base["H"]
    psi1 = base["psi"]
    qb, _, _, _ = S.ols([[Dv[i]] + list(W[i]) for i in range(n)], yv)
    Qobs = [C.dot([Dv[i]] + list(W[i]), qb) for i in range(n)]
    resid = [yv[i] - Qobs[i] - base["eps"] * H[i] for i in range(n)]

    phi = [[1.0] + list(Xm[i]) + [v * v for v in Xm[i]] for i in range(n)]
    k = len(phi[0])
    Om = [[sum(phi[i][a] * phi[i][b] for i in range(n)) / n + (1e-8 if a == b else 0.0)
           for b in range(k)] for a in range(k)]
    Oi = C.inv(Om)
    a = [(Dv[i] - g[i]) / (g[i] * (1.0 - g[i])) for i in range(n)]
    b = [H[i] * resid[i] for i in range(n)]
    u = [[sum(Oi[r][c] * phi[i][c] for c in range(k)) for r in range(k)] for i in range(n)]
    tot = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tot += a[i] * C.dot(phi[i], u[j]) * b[j]
    if22 = -tot / (n * (n - 1))
    est = psi1 + if22
    ic = [base["ic"][i] for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": est, "se": se, "psi1": psi1, "if22": if22,
        "k_basis": float(k), "n": n,
        "method": "TMLE with an estimated second-order remainder"})


def cheatsheet():
    return "tmlres: TMLE with a second-order (residual) bias correction."
