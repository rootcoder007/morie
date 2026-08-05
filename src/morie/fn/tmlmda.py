# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the ATE with a missing-at-random outcome."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_missing_data"]


def tmle_missing_data(y, D, X, missing):
    """Targeted ATE when the outcome is observed only for some subjects.

    Missingness is a second intervention node.  Writing ``Delta = 1`` for
    an observed outcome, the joint intervention is "set ``A = a`` and
    ``Delta = 1``", so the clever covariate carries both probabilities,

        ``H = Delta / pi(A, X) * [A/g(X) - (1 - A)/(1 - g(X))]``,

    with ``pi(a, X) = P(Delta = 1 | A = a, X)`` evaluated at the
    counterfactual arm, not at the observed one.  The initial outcome
    regression is fitted on complete cases only -- under MAR that fit is
    consistent for ``E[Y | A, X]`` -- and the fluctuation
    ``Q*(a, X) = Q(a, X) + eps / (g_a(X) pi(a, X))`` uses the arm-specific
    denominator so that the plug-in solves the joint efficient score.
    With no missingness ``pi == 1`` and the estimator collapses exactly
    onto the standard point-treatment TMLE.

    Determinism: fixed-iteration IRLS for both nuisance models, and a
    closed-form linear fluctuation.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome; entries with ``missing = 1`` are ignored and may be any
        finite placeholder.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.
    missing : array-like, shape (n,)
        1 if the outcome is missing, 0 if observed.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_obs``, ``n``.

    References
    ----------
    Rotnitzky, A., Robins, J. M. & Scharfstein, D. O. (1998).
    Semiparametric regression for repeated outcomes with nonignorable
    nonresponse.  Journal of the American Statistical Association
    93(444):1321-1339.  doi:10.1080/01621459.1998.10473795.  The doubly
    robust plug-in form is Bang, H. & Robins, J. M. (2005), Doubly
    robust estimation in missing data and causal inference models,
    Biometrics 61(4):962-973.  doi:10.1111/j.1541-0420.2005.00377.x.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    mv = C.vec(missing)
    n = len(yv)
    if n == 0 or len(Dv) != n or len(mv) != n:
        raise ValueError("tmle_missing_data: y, D and missing must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_missing_data: X must have one row per subject")
    delta = [1.0 - v for v in mv]
    if sum(delta) < 2:
        raise ValueError("tmle_missing_data: fewer than two observed outcomes")
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    gb = S.glmbin(W, Dv)
    g = [S.clip(S.expit(C.dot(W[i], gb)), 0.025, 0.975) for i in range(n)]
    pdes = [[Dv[i]] + list(W[i]) for i in range(n)]
    pb = S.glmbin(pdes, delta)

    def pihat(i, a):
        return S.clip(S.expit(C.dot([a] + list(W[i]), pb)), 0.025, 1.0)

    obs = [i for i in range(n) if delta[i] > 0.5]
    des = [[Dv[i]] + list(W[i]) for i in obs]
    qb, _, _, _ = S.ols(des, [yv[i] for i in obs])

    def qhat(i, a):
        return C.dot([a] + list(W[i]), qb)

    Q1 = [qhat(i, 1.0) for i in range(n)]
    Q0 = [qhat(i, 0.0) for i in range(n)]
    Qobs = [qhat(i, Dv[i]) for i in range(n)]
    pi_obs = [pihat(i, Dv[i]) for i in range(n)]
    H = [delta[i] / pi_obs[i] * (Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i])) for i in range(n)]
    den = sum(h * h for h in H)
    num = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n) if delta[i] > 0.5)
    eps = num / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / (g[i] * pihat(i, 1.0)) for i in range(n)]
    Q0s = [Q0[i] - eps / ((1.0 - g[i]) * pihat(i, 0.0)) for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = []
    for i in range(n):
        resid = (yv[i] - Qobs[i] - eps * H[i]) if delta[i] > 0.5 else 0.0
        ic.append(H[i] * resid + Q1s[i] - Q0s[i] - psi)
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps,
        "n_obs": float(sum(delta)), "n": n,
        "method": "TMLE for the ATE under a missing-at-random outcome"})


def cheatsheet():
    return "tmlmda: TMLE for the ATE under a missing-at-random outcome."
