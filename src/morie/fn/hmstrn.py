# morie.fn -- function file (rootcoder007/morie)
"""History-adjusted marginal structural model."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["history_adjusted_msm"]


def history_adjusted_msm(y, treatment_history, covariate_history, time, regime):
    """Effect of a treatment rule read off from each time point onwards.

    A standard marginal structural model answers one question at
    baseline.  The history-adjusted version answers it again at every
    time point, conditioning on what was known then, which is what a
    clinician actually needs -- the decision at month six is made with
    six months of information, not none.  Each time-specific model is
    fitted on the units still consistent with the regime, weighted by
    the inverse probability of having followed it.

    Formula: ``E[Y(d) | H_t] = gamma_0 + gamma_1 t``, fitted with
    weights ``prod_s 1 / P(A_s | H_s)`` over the units whose treatment
    history agrees with ``regime``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    treatment_history : array-like, shape (n, T)
        Treatment actually received at each time.
    covariate_history : array-like, shape (n, T)
        Time-varying covariate.
    time : array-like, shape (T,)
        Time points.
    regime : array-like, shape (T,)
        Treatment the rule prescribes at each time.

    Returns
    -------
    RichResult
        ``estimate`` (slope in time of the regime-specific mean),
        ``intercept``, ``by_time``, ``n_consistent``, ``n``.

    References
    ----------
    van der Laan, M. J. & Petersen, M. L. (2007).  Causal effect models
    for realistic individualized treatment and intention to treat
    rules.  International Journal of Biostatistics 3(1):3.  The
    history-adjusted MSM is van der Laan, Petersen & Joffe (2005), IJB
    1(1):4.
    """
    yv = C.vec(y)
    A = C.mat(treatment_history)
    L = C.mat(covariate_history)
    tv = C.vec(time)
    d = C.vec(regime)
    n, T = C.shape(A)
    means, times = [], []
    for t in range(T):
        idx = [i for i in range(n) if all(abs(A[i][s] - d[s]) < 0.5 for s in range(t + 1))]
        if not idx:
            continue
        des = C.cbind1([[L[i][t]] for i in idx])
        gb = S.glmbin(des, [A[i][t] for i in idx])
        w = []
        for k, i in enumerate(idx):
            g = S.clip(S.expit(C.dot(des[k], gb)), 0.025, 0.975)
            w.append(1.0 / g if A[i][t] > 0.5 else 1.0 / (1.0 - g))
        sw = sum(w)
        means.append(sum(w[k] * yv[idx[k]] for k in range(len(idx))) / sw)
        times.append(tv[t])
    des = [[1.0, t] for t in times]
    beta, _, _, _ = S.ols(des, means)
    idx0 = [i for i in range(n) if all(abs(A[i][s] - d[s]) < 0.5 for s in range(T))]
    return RichResult(payload={
        "estimate": beta[1], "intercept": beta[0], "by_time": means,
        "n_consistent": len(idx0), "n": n,
        "method": "History-adjusted marginal structural model"})


def cheatsheet():
    return "hmstrn: History-adjusted marginal structural model."
