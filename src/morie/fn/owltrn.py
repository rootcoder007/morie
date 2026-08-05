# morie.fn -- function file (rootcoder007/morie)
"""Outcome-weighted learning for an optimal treatment regime."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["outcome_weighted_learning"]


def outcome_weighted_learning(y, D, W, pi=None, lam=0.01, n_iter=2000):
    """Individualized treatment rule by outcome-weighted classification.

    Zhao et al.'s reformulation is that finding the regime maximizing
    the value

        V(d) = E[ Y 1{A = d(X)} / pi(A | X) ]

    is the same problem as MINIMIZING the weighted misclassification
    risk

        E[ (Y / pi(A | X)) 1{A != d(X)} ],

    a classification problem in which the outcome is the weight and the
    observed treatment is the label.  The surrogate is theirs -- the
    weighted HINGE loss with a ridge penalty, which is a linear-kernel
    weighted SVM,

        min_b  (1/n) sum_i w_i max(0, 1 - A_i x_i'b) + lam ||b_slope||^2,

    minimized by full-batch subgradient descent on the Pegasos step
    schedule ``eta_t = 1 / (lam t)``: a fixed iteration count, no
    sampling and no tolerance test, so both language arms take exactly
    the same path.  The intercept is not penalized.

    The cheaper surrogate -- weighted least squares of the label on the
    covariates -- has the right population minimiser but is not robust
    to leverage.  On a design where the high-weight units sit near the
    decision boundary and the low-weight ones sit far from it, the
    squared loss lets the far units outvote the informative ones and
    returns the exactly INVERTED rule.  That happened on the fixture in
    this module's own anchors, which is why the hinge is used.

    The rule is ``d(x) = 1{x'b > 0}``.  NEGATIVE outcomes break the
    equivalence, because a negative weight rewards misclassification;
    the standard fix is to shift ``Y`` by its minimum.  That shift is
    NOT innocuous -- it changes the relative weights and so can change
    the learned rule, which is exactly the objection that motivated
    residual weighted learning -- so the amount shifted is returned in
    ``shift`` rather than applied silently.

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
        for a completely randomized trial.
    lam : float, default 0.01
        Ridge penalty on the slope coefficients, positive.
    n_iter : int, default 2000
        Subgradient iterations.

    Returns
    -------
    RichResult
        ``beta``, ``estimate`` (the estimated value of the learned
        rule), ``value``, ``value_all_treated``, ``value_all_control``,
        ``rule`` (recommended treatment per unit, 0/1),
        ``n_treated_by_rule``, ``hinge`` (the objective at the
        solution), ``shift``, ``n``, ``p``.

    References
    ----------
    Zhao, Y., Zeng, D., Rush, A. J. & Kosorok, M. R. (2012).
    Estimating individualized treatment rules using outcome weighted
    learning.  Journal of the American Statistical Association,
    107(499), 1106--1118.  doi:10.1080/01621459.2012.695674
    Shalev-Shwartz, S., Singer, Y., Srebro, N. & Cotter, A. (2011).
    Pegasos: primal estimated sub-gradient solver for SVM.
    Mathematical Programming, 127(1), 3--30.
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
    lm = float(lam)
    if lm <= 0.0:
        raise ValueError("outcome_weighted_learning: lam must be positive")
    T = int(n_iter)
    if T < 1:
        raise ValueError("outcome_weighted_learning: n_iter must be at least 1")

    ymin = min(yv)
    shift = -ymin if ymin < 0.0 else 0.0
    ys = [v + shift for v in yv]
    w = [ys[i] / pv[i] for i in range(n)]
    wbar = sum(w) / n
    if wbar <= 0.0:
        raise ValueError("outcome_weighted_learning: every weight is zero")
    w = [v / wbar for v in w]
    lab = [1.0 if Dv[i] == 1.0 else -1.0 for i in range(n)]

    beta = [0.0] * p
    for t in range(1, T + 1):
        eta = 1.0 / (lm * t)
        gr = [0.0] * p
        for k in range(1, p):
            gr[k] = lm * beta[k]
        for i in range(n):
            marg = lab[i] * sum(Wm[i][k] * beta[k] for k in range(p))
            if marg < 1.0:
                for k in range(p):
                    gr[k] -= w[i] * lab[i] * Wm[i][k] / n
        for k in range(p):
            beta[k] -= eta * gr[k]

    hinge = sum(w[i] * max(0.0, 1.0 - lab[i]
                           * sum(Wm[i][k] * beta[k] for k in range(p)))
                for i in range(n)) / n
    hinge += lm * sum(beta[k] * beta[k] for k in range(1, p))
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
        "n_treated_by_rule": float(sum(rule)), "hinge": hinge,
        "shift": shift, "n": n, "p": p,
        "method": "Outcome-weighted learning, weighted hinge (Zhao et al. 2012)"})


def cheatsheet():
    return "owltrn: Outcome-weighted learning for an optimal regime"


outcomeweightedlearning = outcome_weighted_learning
