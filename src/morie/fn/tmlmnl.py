# morie.fn -- function file (rootcoder007/morie)
"""Cross-fitted TMLE that accepts arbitrary machine learners for Q and g."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_machine_learning"]


def _fit_q(Xtr, ytr, Xte):
    """Default outcome learner: least squares through the shared QR."""
    b, _, _, _ = S.ols(Xtr, ytr)
    return [C.dot(row, b) for row in Xte]


def _fit_g(Xtr, dtr, Xte):
    """Default propensity learner: fixed-iteration logistic IRLS."""
    b = S.glmbin(Xtr, dtr)
    return [S.expit(C.dot(row, b)) for row in Xte]


def tmle_machine_learning(y, D, X, ml_q=None, ml_g=None):
    """TMLE with cross-fitted nuisances from user-supplied learners.

    Cross-fitting is what makes an arbitrarily complex learner safe here:
    each observation's nuisance value is produced by a fit that never saw
    it, so the empirical-process term in the expansion vanishes without
    a Donsker condition on the learner.  This is the debiased-machine-
    learning argument of Chernozhukov et al. (2018); the targeting step
    is the TMLE one, so the estimator is a plug-in and stays in the
    parameter space.

    Formula: with out-of-fold ``Q`` and ``g``,
    ``H = D/g - (1 - D)/(1 - g)``, ``eps = sum H (y - Q)/sum H^2``,
    ``psi = mean[Q*(1, X) - Q*(0, X)]``.

    Folds are ``i mod 5`` on the input order -- deterministic by
    construction, so both language arms cross-fit on exactly the same
    partition.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment.
    X : array-like, shape (n, p)
        Covariates.
    ml_q : callable or None
        ``ml_q(Xtrain, ytrain, Xtest) -> predictions``.  ``Xtrain`` and
        ``Xtest`` carry an intercept and the treatment column first.
        ``None`` uses least squares.
    ml_g : callable or None
        ``ml_g(Xtrain, Dtrain, Xtest) -> probabilities``.  ``None`` uses
        logistic IRLS.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``n_folds``, ``n``.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
    C., Newey, W. & Robins, J. (2018).  Double/debiased machine learning
    for treatment and structural parameters.  Econometrics Journal
    21(1):C1-C68.  doi:10.1111/ectj.12097.  The targeting step is van
    der Laan, M. J. & Rubin, D. (2006), IJB 2(1):11.
    """
    yv = C.vec(y)
    Dv = C.vec(D)
    n = len(yv)
    if n == 0 or len(Dv) != n:
        raise ValueError("tmle_machine_learning: y and D must share one length")
    Xm = C.mat(X)
    if len(Xm) != n:
        raise ValueError("tmle_machine_learning: X must have one row per subject")
    fq = _fit_q if ml_q is None else ml_q
    fg = _fit_g if ml_g is None else ml_g
    K = 5
    W = [[1.0] + list(Xm[i]) for i in range(n)]
    des = [[Dv[i]] + list(W[i]) for i in range(n)]
    g = [0.0] * n
    Qobs = [0.0] * n
    Q1 = [0.0] * n
    Q0 = [0.0] * n
    for k in range(K):
        te = [i for i in range(n) if i % K == k]
        tr = [i for i in range(n) if i % K != k]
        if not te or len(tr) < 2:
            continue
        gp = fg([W[i] for i in tr], [Dv[i] for i in tr], [W[i] for i in te])
        for j, i in enumerate(te):
            g[i] = S.clip(float(gp[j]), 0.025, 0.975)
        stack = [des[i] for i in te] + [[1.0] + list(W[i]) for i in te] + \
                [[0.0] + list(W[i]) for i in te]
        qp = fq([des[i] for i in tr], [yv[i] for i in tr], stack)
        m = len(te)
        for j, i in enumerate(te):
            Qobs[i] = float(qp[j])
            Q1[i] = float(qp[m + j])
            Q0[i] = float(qp[2 * m + j])
    H = [Dv[i] / g[i] - (1.0 - Dv[i]) / (1.0 - g[i]) for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g[i] for i in range(n)]
    Q0s = [Q0[i] - eps / (1.0 - g[i]) for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Q1s[i] - Q0s[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "n_folds": float(K), "n": n,
        "method": "Cross-fitted TMLE with pluggable machine learners"})


def cheatsheet():
    return "tmlmnl: cross-fitted TMLE with arbitrary learners for Q and g."
