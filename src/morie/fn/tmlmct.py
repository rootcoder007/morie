# morie.fn -- function file (rootcoder007/morie)
"""TMLE for a vector-valued (multivariate) binary treatment."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tmle_multivariate_treatment"]


def tmle_multivariate_treatment(y, A, X):
    """Targeted contrast of ``E[Y(1,...,1)]`` against ``E[Y(0,...,0)]``.

    With a treatment vector the propensity is not a single logistic; it
    is the sequential factorisation

        ``g(a | X) = prod_j P(A_j = a_j | X, A_1 = a_1, ..., A_{j-1} = a_{j-1})``

    fitted by one logistic per component with the preceding components
    as regressors.  The point that matters is that the counterfactual
    probability is obtained by evaluating those SAME fitted coefficients
    at the counterfactual history ``a_{<j}``, not by refitting: refitting
    would condition on the observed history and silently target a
    different parameter.

    Given ``g``, the machinery is the point-treatment one with the
    indicator on the whole vector,

        ``H = I(A = 1)/g(1|X) - I(A = 0)/g(0|X)``,
        ``eps = sum H (y - Q) / sum H^2``,
        ``psi = mean[Q*(1, X) - Q*(0, X)]``.

    With a single component this reduces exactly to the binary
    point-treatment TMLE.

    Determinism: fixed-iteration IRLS per component, closed-form linear
    fluctuation.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    A : array-like, shape (n, q)
        Binary treatment vector, one column per component.
    X : array-like, shape (n, p)
        Covariates.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``eps``, ``q``, ``n``.

    References
    ----------
    Lendle, S. D., Schwab, J., Petersen, M. L. & van der Laan, M. J.
    (2017).  ltmle: an R package implementing targeted minimum
    loss-based estimation for longitudinal data.  Journal of Statistical
    Software 81(1).  doi:10.18637/jss.v081.i01.  The sequential
    factorisation of a joint intervention is Robins, J. M. (1986), A new
    approach to causal inference in mortality studies, Mathematical
    Modelling 7:1393-1512.
    """
    yv = C.vec(y)
    n = len(yv)
    Am = C.mat(A)
    Xm = C.mat(X)
    if n == 0 or len(Am) != n or len(Xm) != n:
        raise ValueError("tmle_multivariate_treatment: y, A and X must share n rows")
    q = len(Am[0])
    Wd = [[1.0] + list(Xm[i]) for i in range(n)]

    betas = []
    for j in range(q):
        des = [Wd[i] + [Am[i][k] for k in range(j)] for i in range(n)]
        betas.append(S.glmbin(des, [Am[i][j] for i in range(n)]))

    def gprob(i, a):
        p = 1.0
        for j in range(q):
            z = C.dot(Wd[i] + [a] * j, betas[j])
            pj = S.clip(S.expit(z), 1e-6, 1.0 - 1e-6)
            p *= pj if a > 0.5 else 1.0 - pj
        return S.clip(p, 0.01, 0.99)

    g1 = [gprob(i, 1.0) for i in range(n)]
    g0 = [gprob(i, 0.0) for i in range(n)]
    des = [[Am[i][j] for j in range(q)] + list(Wd[i]) for i in range(n)]
    qb, _, _, _ = S.ols(des, yv)

    def qhat(i, a):
        return C.dot([a] * q + list(Wd[i]), qb)

    Q1 = [qhat(i, 1.0) for i in range(n)]
    Q0 = [qhat(i, 0.0) for i in range(n)]
    Qobs = [C.dot(des[i], qb) for i in range(n)]
    all1 = [1.0 if all(v > 0.5 for v in Am[i]) else 0.0 for i in range(n)]
    all0 = [1.0 if all(v < 0.5 for v in Am[i]) else 0.0 for i in range(n)]
    H = [all1[i] / g1[i] - all0[i] / g0[i] for i in range(n)]
    den = sum(h * h for h in H)
    eps = sum(H[i] * (yv[i] - Qobs[i]) for i in range(n)) / den if den != 0.0 else 0.0
    Q1s = [Q1[i] + eps / g1[i] for i in range(n)]
    Q0s = [Q0[i] - eps / g0[i] for i in range(n)]
    psi = sum(Q1s[i] - Q0s[i] for i in range(n)) / n
    ic = [H[i] * (yv[i] - Qobs[i] - eps * H[i]) + Q1s[i] - Q0s[i] - psi for i in range(n)]
    m = sum(ic) / n
    se = math.sqrt(sum((v - m) ** 2 for v in ic) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": psi, "se": se, "eps": eps, "q": float(q), "n": n,
        "method": "TMLE for a vector-valued binary treatment"})


def cheatsheet():
    return "tmlmct: TMLE for a vector-valued (multivariate) treatment."
