# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the marginal probabilistic index P(Y(1) > Y(0))."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_marginal_pim"]


def _cdf_bank(yv, Dv, W, g, grid):
    """Targeted counterfactual CDFs and their influence curves on a grid.

    One point-treatment TMLE per threshold, on the binary outcome
    ``I(Y <= t)``, with an arm-specific clever covariate
    ``H_a = I(D = a)/g_a`` so that each arm mean -- not just the
    contrast -- solves its own efficient score.  The targeted CDFs are
    made monotone by a running maximum, which is the cheapest
    rearrangement that cannot move a correctly ordered pair.
    """
    n = len(yv)
    K = len(grid)
    F = [[0.0] * K, [0.0] * K]
    IC = [[[0.0] * n for _ in range(K)], [[0.0] * n for _ in range(K)]]
    for j in range(K):
        z = [1.0 if yv[i] <= grid[j] else 0.0 for i in range(n)]
        des = [[Dv[i]] + list(W[i]) for i in range(n)]
        qb, _, _, _ = S.ols(des, z)
        Q = [[C.dot([0.0] + list(W[i]), qb) for i in range(n)],
             [C.dot([1.0] + list(W[i]), qb) for i in range(n)]]
        for a in (0, 1):
            ga = [g[i] if a == 1 else 1.0 - g[i] for i in range(n)]
            H = [(1.0 if abs(Dv[i] - a) < 0.5 else 0.0) / ga[i] for i in range(n)]
            den = sum(h * h for h in H)
            eps = sum(H[i] * (z[i] - Q[a][i]) for i in range(n)) / den if den != 0.0 else 0.0
            Qs = [S.clip(Q[a][i] + eps / ga[i], 0.0, 1.0) for i in range(n)]
            p = sum(Qs) / n
            F[a][j] = p
            for i in range(n):
                IC[a][j][i] = H[i] * (z[i] - Q[a][i] - eps * H[i]) + Qs[i] - p
    for a in (0, 1):
        run = 0.0
        for j in range(K):
            run = max(run, F[a][j])
            F[a][j] = min(run, 1.0)
    return F, IC


def tmle_marginal_pim(y, D, X):
    """Targeted estimate of ``P(Y(1) > Y(0)) + 0.5 P(Y(1) = Y(0))``.

    The probabilistic index is a functional of the two counterfactual
    marginal distributions, not of a regression coefficient, so it is
    built from a bank of threshold TMLEs: for every distinct observed
    outcome value the binary parameter ``F_a(t) = P(Y(a) <= t)`` is
    targeted, and the index is the Riemann-Stieltjes integral

        ``psi = sum_j [ (F_0(t_{j-1}) + F_0(t_j)) / 2 ] dF_1(t_j)``,

    whose mid-point weight is exactly the half-credit-for-ties
    convention.  Because ``Y(1)`` and ``Y(0)`` enter only through their
    marginals this is the MARGINAL probabilistic index; it does not
    require a joint model and is not the within-pair index.

    The influence curve is the delta-method combination of the threshold
    influence curves, so the reported SE accounts for the whole bank.

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
        ``estimate``, ``se``, ``n_grid``, ``n``.

    References
    ----------
    Thas, O., De Neve, J., Clement, L. & Ottoy, J. P. (2012).
    Probabilistic index models.  Journal of the Royal Statistical
    Society Series B 74(4):623-671.
    doi:10.1111/j.1467-9868.2011.01020.x.  The targeting step is van der
    Laan, M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_marginal_pim: y and D must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_marginal_pim: X must have one row per subject")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    grid = sorted(set(yv))
    K = len(grid)
    F, IC = _cdf_bank(yv, Dv, W, g, grid)
    psi = 0.0
    ic = [0.0] * n
    for j in range(K):
        f0p = F[0][j - 1] if j > 0 else 0.0
        f1p = F[1][j - 1] if j > 0 else 0.0
        bar = 0.5 * (f0p + F[0][j])
        d1 = F[1][j] - f1p
        psi += bar * d1
        for i in range(n):
            prev1 = IC[1][j - 1][i] if j > 0 else 0.0
            prev0 = IC[0][j - 1][i] if j > 0 else 0.0
            ic[i] += bar * (IC[1][j][i] - prev1) + 0.5 * (IC[0][j][i] + prev0) * d1
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "n_grid": float(K), "n": n,
        "method": "TMLE for the marginal probabilistic index"})


def cheatsheet():
    return "tmlmpi: TMLE for the marginal probabilistic index."
