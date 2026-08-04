# morie.fn -- function file (rootcoder007/morie)
"""TMLE targeting a pre-treatment outcome model."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_baseline_adj"]


def tmle_baseline_adj(y, D, X, baseline):
    """TMLE whose initial outcome model uses baseline covariates only.

    The point of restricting the initial fit to pre-treatment
    information is that nothing measured after randomisation can leak
    into ``Q``, so the targeting step is doing the whole job of removing
    residual imbalance rather than sharing it with a post-treatment
    adjustment that would open a collider path.  The clever covariate
    carries the propensity, so the estimator stays consistent when
    either the outcome model or the propensity model is right.

    Determinism: propensity by IRLS at a fixed iteration count, and a
    closed-form linear fluctuation, so there is no line search.

    Formula: ``H = D / g - (1 - D) / (1 - g)``,
    ``eps = sum H (y - Q) / sum H^2``, ``Q* = Q + eps H``, and
    ``psi = mean[Q*(1, W) - Q*(0, W)]`` for ``W = [X, baseline]``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Pre-treatment covariates.
    baseline : array-like, shape (n,)
        Baseline level of the outcome, also pre-treatment.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n``.

    References
    ----------
    Tsiatis, A. A., Davidian, M., Zhang, M. & Lu, X. (2008).  Covariate
    adjustment for two-sample treatment comparisons in randomized
    clinical trials.  Statistics in Medicine 27:4658-4677.  The
    targeting step is van der Laan & Rubin (2006), Targeted maximum
    likelihood learning, International Journal of Biostatistics 2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    bl = C.vec(baseline)
    n = len(yv)
    W = [[1.0] + list(C.mat(X)[i]) + [bl[i]] for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    des = [[Dv[i]] + list(W[i]) for i in range(n)]
    qb, _, _, _ = C.lstsq(des, yv)
    def qhat(d):
        return [C.dot([d] + list(W[i]), qb) for i in range(n)]
    Q1 = qhat(1.0)
    Q0 = qhat(0.0)
    Q = [Q1[i] if Dv[i] > 0.5 else Q0[i] for i in range(n)]
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Q[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    Qs = [Q[i] + eps * H[i] for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = [H[i] * (yv[i] - Qs[i]) + Q1s[i] - Q0s[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "n": n,
        "method": "TMLE with a pre-treatment-only initial outcome model"})


def cheatsheet():
    return "tmlbas: TMLE targeting a pre-treatment outcome model."
