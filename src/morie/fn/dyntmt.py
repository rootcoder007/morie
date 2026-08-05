# morie.fn -- function file (rootcoder007/morie)
"""Value of a dynamic treatment regime under a marginal structural model.

Sources: Robins, J. M., Orellana, L. and Rotnitzky, A. (2008).
Estimation and extrapolation of optimal treatment and testing
strategies.  *Statistics in Medicine* 27(23), 4678-4721,
doi:10.1002/sim.3301, which sets up marginal structural models indexed
by a family of dynamic regimes d and defines the value

    V(d) = E[Y(d_bar(H))],

the mean counterfactual outcome had every subject been treated
according to the rule d applied to their own evolving history H; and
Petersen, M., Schwab, J., Gruber, S., Blaser, N., Schomaker, M. and van
der Laan, M. (2014).  Targeted maximum likelihood estimation for dynamic
and static longitudinal marginal structural working models.  *Journal of
Causal Inference* 2(2), 147-185, doi:10.1515/jci-2013-0007, which fits
the same working model by TMLE.

Estimated here by the inverse-probability weighted (Hajek) form, which
is the estimator Robins, Orellana and Rotnitzky give first and against
which their augmented estimator is defined:

    C_i     = 1 if the observed treatment history follows d at every t
    SW_i    = prod_t P(D_it = d_it) / P(D_it = d_it | H_it)
    V_hat(d) = sum_i C_i SW_i Y_i / sum_i C_i SW_i.

The numerator probability is the marginal treatment frequency at time
t, which is what makes the weights *stabilised*; the denominator is a
pooled logistic fit of treatment on the history, so the weights are
deterministic given the data.  A rule that every subject happens to
follow makes C identically one and the stabilised weights collapse to
one, at which point V_hat is the plain sample mean of Y -- the
degenerate identity this module is anchored on.
"""

from __future__ import annotations

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dynamic_marginal_msm"]


def dynamic_marginal_msm(y, D_history, H_history, regime_fn=None):
    """IPTW value of a dynamic regime, with stabilised weights.

    Parameters
    ----------
    y : array-like
        End-of-follow-up outcome, one entry per subject.
    D_history : 2-D array-like
        Observed binary treatment, subjects by time points.
    H_history : 2-D array-like
        Time-varying history used by the rule and by the propensity
        fit, subjects by time points.
    regime_fn : callable or array-like, optional
        The rule.  A callable is applied elementwise to ``H_history``
        and must return 0/1.  A 2-D array is taken as the prescribed
        treatment directly.  ``None`` prescribes the rule "treat once
        the history exceeds zero", the canonical threshold regime.

    Returns
    -------
    result : dict
        Keys: estimate (V(d)), value, n_follow, sum_weights,
        mean_weight, max_weight, ess, naive_mean, n_time, n.

    References
    ----------
    Robins, Orellana & Rotnitzky (2008), Statist. Med. 27(23):4678-4721,
    doi:10.1002/sim.3301.
    Petersen, Schwab, Gruber, Blaser, Schomaker & van der Laan (2014),
    J. Causal Inference 2(2):147-185, doi:10.1515/jci-2013-0007.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    D = core.mat(D_history)
    H = core.mat(H_history)
    if len(D) != n or len(H) != n:
        raise ValueError("D_history and H_history must have one row per subject")
    T = len(D[0])
    if T == 0:
        raise ValueError("D_history has no time points")
    for i in range(n):
        if len(D[i]) != T or len(H[i]) != T:
            raise ValueError("ragged history: every row must have T entries")
        for v in D[i]:
            if v != 0.0 and v != 1.0:
                raise ValueError("D_history must be binary 0/1")
    if regime_fn is None:
        pres = [[1.0 if H[i][t] > 0.0 else 0.0 for t in range(T)]
                for i in range(n)]
    elif callable(regime_fn):
        pres = [[1.0 if regime_fn(H[i][t]) else 0.0 for t in range(T)]
                for i in range(n)]
    else:
        R = core.mat(regime_fn)
        if len(R) != n or len(R[0]) != T:
            raise ValueError("regime_fn as an array must be n x T")
        pres = [[1.0 if R[i][t] > 0.5 else 0.0 for t in range(T)]
                for i in range(n)]
    follow = [1.0] * n
    for i in range(n):
        for t in range(T):
            if D[i][t] != pres[i][t]:
                follow[i] = 0.0
                break
    w = [1.0] * n
    for t in range(T):
        dt = [D[i][t] for i in range(n)]
        marg = core.mean(dt)
        if marg <= 0.0 or marg >= 1.0:
            continue
        Z = [[1.0, H[i][t]] for i in range(n)]
        gam = core.logit_irls(Z, dt)
        for i in range(n):
            p = core.sigmoid(Z[i][0] * gam[0] + Z[i][1] * gam[1])
            if p < 1e-12:
                p = 1e-12
            elif p > 1.0 - 1e-12:
                p = 1.0 - 1e-12
            num = marg if pres[i][t] > 0.5 else (1.0 - marg)
            den = p if pres[i][t] > 0.5 else (1.0 - p)
            w[i] *= num / den
    num, den = 0.0, 0.0
    for i in range(n):
        num += follow[i] * w[i] * yv[i]
        den += follow[i] * w[i]
    if den <= 0.0:
        raise ValueError("no subject follows the regime: V(d) is not identified")
    ww = [follow[i] * w[i] for i in range(n)]
    s1 = sum(ww)
    s2 = sum(v * v for v in ww)
    return RichResult(
        title="Dynamic-regime MSM value",
        summary_lines=[("followers", sum(follow))],
        payload={
            "estimate": num / den,
            "value": num / den,
            "n_follow": sum(follow),
            "sum_weights": s1,
            "mean_weight": s1 / n,
            "max_weight": max(ww),
            "ess": (s1 * s1 / s2) if s2 > 0.0 else 0.0,
            "naive_mean": core.mean(yv),
            "n_time": float(T),
            "n": n,
            "method": "Dynamic-regime MSM (regime depends on history)",
        },
    )


def cheatsheet():
    return "dyntmt: Dynamic-regime MSM (regime depends on history)"
