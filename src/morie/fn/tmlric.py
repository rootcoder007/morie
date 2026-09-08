# morie.fn -- function file (rootcoder007/morie)
"""Case-control-weighted TMLE for a rare outcome."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_rare_outcome"]


def tmle_rare_outcome(y, D, X, prevalence):
    """Targeted maximum likelihood for a rare binary outcome.

    When the outcome is rare, a sample drawn on the outcome carries no
    information about how rare it is, so the estimator has to be told.
    Known-prevalence case-control weighting supplies that: cases are
    down- or up-weighted to ``q0`` and controls to ``1 - q0``, and the
    weights enter both the initial fit and the targeting step, which is
    what keeps the estimator double robust rather than merely
    consistent under a correct outcome model.

    Determinism: the propensity model is IRLS with a fixed iteration
    count and the fluctuation is the closed-form linear one, so there
    is no line search and no convergence tolerance anywhere.

    Formula: ``H(D, X) = D / g(X) - (1 - D) / (1 - g(X))``;
    ``Q*(D, X) = Q(D, X) + eps H(D, X)`` with
    ``eps = sum_i w_i H_i (y_i - Q_i) / sum_i w_i H_i^2``;
    ``psi = sum_i w_i [Q*(1, X_i) - Q*(0, X_i)] / sum_i w_i``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Binary outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Baseline covariates; an intercept is added.
    prevalence : float
        Known population prevalence ``q0`` of ``y = 1``.

    Returns
    -------
    RichResult
        ``estimate`` (weighted ATE), ``se``, ``eps``, ``n``.

    References
    ----------
    Tran, L., Petersen, M., Schwab, J. & van der Laan, M. J. (2018).
    Robust variance estimation and inference for causal effect
    estimation.  arXiv:1810.03030.  The case-control weighting scheme
    is van der Laan (2008), Estimation based on case-control designs
    with known prevalence probability, IJB 4(1):17, whose weights
    ``q0 n / n1`` and ``(1 - q0) n / n0`` are used verbatim.
    """
    y = C.vec(y)
    D = C.vec(D)
    n = len(y)
    Xm = C.cbind1(C.mat(X)) if not isinstance(X[0], (int, float)) else C.cbind1([[v] for v in C.vec(X)])
    q0 = float(prevalence)
    n1 = sum(1 for v in y if v > 0.5)
    n0 = n - n1
    w = [(q0 * n / n1 if v > 0.5 else (1.0 - q0) * n / n0) for v in y]
    gdes = Xm
    gb = S.glmbin(gdes, D)
    g = [S.clip(S.expit(C.dot(gdes[i], gb)), 0.01, 0.99) for i in range(n)]
    qdes = [[D[i]] + list(Xm[i]) for i in range(n)]
    qb = S.glmbin(qdes, y)
    def qhat(d, i):
        row = [d] + list(Xm[i])
        return S.clip(S.expit(C.dot(row, qb)), 1e-6, 1.0 - 1e-6)
    Q = [qhat(D[i], i) for i in range(n)]
    Q1 = [qhat(1.0, i) for i in range(n)]
    Q0 = [qhat(0.0, i) for i in range(n)]
    H = [D[i] / g[i] - (1.0 - D[i]) / (1.0 - g[i]) for i in range(n)]
    num = sum(w[i] * H[i] * (y[i] - Q[i]) for i in range(n))
    den = sum(w[i] * H[i] * H[i] for i in range(n))
    eps = num / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    Qs = [Q[i] + eps * H[i] for i in range(n)]
    sw = sum(w)
    psi = sum(w[i] * (Q1s[i] - Q0s[i]) for i in range(n)) / sw
    ic = [w[i] * (H[i] * (y[i] - Qs[i]) + Q1s[i] - Q0s[i] - psi) / (sw / n) for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "n": n,
        "method": "Case-control-weighted TMLE, rare outcome"})


def cheatsheet():
    return "tmlric: Case-control-weighted TMLE for a rare outcome."
