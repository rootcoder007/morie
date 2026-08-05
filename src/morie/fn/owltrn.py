# morie.fn -- function file (rootcoder007/morie)
"""Outcome-weighted learning for an optimal treatment regime."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["outcome_weighted_learning"]


def outcome_weighted_learning(y, D, W, pi=None):
    """Individualized treatment rule by outcome-weighted classification.

    Zhao et al.'s reformulation is that finding the regime maximizing
    the value

        V(d) = E[ Y 1{A = d(X)} / pi(A | X) ]

    is the same problem as MINIMIZING the weighted misclassification
    risk

        E[ (Y / pi(A | X)) 1{A != d(X)} ],

    a classification problem in which the outcome is the weight and the
    observed treatment is the label.  Any classifier will do; the
    surrogate used here is weighted least squares of the +/-1 treatment
    label on the covariates, which is the least-squares surrogate of
    the same risk and keeps the fit deterministic and closed form -- no
    kernel, no solver tolerance, no random restarts.

    The decision rule is ``d(x) = sign(x'b)``, mapped back to the
    original treatment coding.  NEGATIVE outcomes break the
    equivalence, because a negative weight rewards misclassification;
    the standard fix, shifting ``Y`` by its minimum so all weights are
    non-negative, is applied and reported rather than done silently, as
    it changes the value scale but not the argmax.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed outcome, larger is better.
    D : array-like, shape (n,)
        Observed binary treatment, coded 0/1.
    W : array-like, shape (n, p)
        Covariates, no intercept column; one is added.
    pi : array-like or None
        Propensity ``P(A = D_i | W_i)`` per unit.  If ``None`` the
        marginal randomization probability is used, which is correct
        for a completed randomized trial.

    Returns
    -------
    RichResult
        ``beta``, ``estimate`` (the estimated value of the learned
        rule), ``value``, ``value_all_treated``, ``value_all_control``,
        ``rule`` (recommended treatment per unit, 0/1),
        ``n_treated_by_rule``, ``shift`` (the constant added to ``y``),
        ``n``, ``p``.

    References
    ----------
    Zhao, Y., Zeng, D., Rush, A. J. & Kosorok, M. R. (2012).
    Estimating individualized treatment rules using outcome weighted
    learning.  Journal of the American Statistical Association,
    107(499), 1106--1118.  doi:10.1080/01621459.2012.695674
    """
    yv = C.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("outcome_weighted_learning: y is empty")
    Dv = C.vec(D)
    if len(Dv) != n:
        raise ValueError("outcome_weighted_learning: y and D have different lengths")
    if any(v not in (0.0, 1.0) for v in Dv):
        raise ValueError("outcome_weighted_learning: D must be binary 0/1")
    Wm = C.cbind1(C.mat(W))
    if len(Wm) != n:
        raise ValueError("outcome_weighted_learning: W and y have different lengths")
    p = len(Wm[0])
    if pi is None:
        pt = sum(Dv) / n
        if pt <= 0.0 or pt >= 1.0:
            raise ValueError("outcome_weighted_learning: both treatments must be observed")
        pv = [pt if Dv[i] == 1.0 else 1.0 - pt for i in range(n)]
    else:
        pv = C.vec(pi)
        if len(pv) != n:
            raise ValueError("outcome_weighted_learning: pi and y have different lengths")
        if any(v <= 0.0 or v > 1.0 for v in pv):
            raise ValueError("outcome_weighted_learning: pi must lie in (0, 1]")
    ymin = min(yv)
    shift = -ymin if ymin < 0.0 else 0.0
    ys = [v + shift for v in yv]
    w = [ys[i] / pv[i] for i in range(n)]
    lab = [1.0 if Dv[i] == 1.0 else -1.0 for i in range(n)]
    # weighted least squares: solve (X' W X) b = X' W lab
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for i in range(n):
        for a in range(p):
            b[a] += Wm[i][a] * w[i] * lab[i]
            for c in range(p):
                A[a][c] += Wm[i][a] * w[i] * Wm[i][c]
    for a in range(p):
        A[a][a] += 1e-10
    beta = C.solvev(A, b)
    rule = [1.0 if sum(Wm[i][k] * beta[k] for k in range(p)) > 0.0 else 0.0
            for i in range(n)]

    def value(rec):
        num = 0.0
        den = 0.0
        for i in range(n):
            m = 1.0 if rec[i] == Dv[i] else 0.0
            num += ys[i] * m / pv[i]
            den += m / pv[i]
        return num / den if den > 0.0 else float("nan")

    return RichResult(payload={
        "beta": beta, "estimate": value(rule), "value": value(rule),
        "value_all_treated": value([1.0] * n),
        "value_all_control": value([0.0] * n), "rule": rule,
        "n_treated_by_rule": float(sum(rule)), "shift": shift,
        "n": n, "p": p,
        "method": "Outcome-weighted learning (Zhao et al. 2012)"})


def cheatsheet():
    return "owltrn: Outcome-weighted learning for an optimal regime"


outcomeweightedlearning = outcome_weighted_learning
