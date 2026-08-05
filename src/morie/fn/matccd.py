# morie.fn -- function file (rootcoder007/morie)
"""Conditional maximum likelihood for matched case-control data."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["matched_case_control"]


def matched_case_control(cases, controls, matching_id, exposure, level=0.95,
                         max_iter=100, tol=1e-12):
    """Estimate the odds ratio without ever estimating the matching sets.

    Matching removes confounding by design, but it introduces one nuisance
    parameter per matched set, and those parameters do not go away as the
    sample grows -- there is always one more per new set.  Ordinary
    logistic regression therefore returns an odds ratio biased away from
    one, badly so for 1:1 pairs.  Conditioning on the number of cases in
    each set cancels the nuisance parameters exactly, leaving a likelihood
    in the exposure effect alone.

    Formula: for a set with one case, ``L_s(beta) = exp(beta x_case) /
    sum_j exp(beta x_j)``; the score and information are the usual
    multinomial ones and the maximum is found by Newton-Raphson --
    Breslow & Day (1980), Chapter 7.

    Parameters
    ----------
    cases : array-like
        Per-observation case indicator, 1 for a case and 0 for a control.
    controls : array-like or None
        Optional control indicator; if supplied it must be ``1 - cases``,
        and it is checked.  Accepted for interface compatibility.
    matching_id : array-like
        Matched-set label per observation.  Exactly one case per set.
    exposure : array-like
        Exposure value per observation; binary or continuous.
    level : float, default 0.95
        Confidence level.
    max_iter : int, default 100
        Newton steps.
    tol : float, default 1e-12
        Convergence tolerance on the score.

    Returns
    -------
    RichResult
        ``estimate`` (the odds ratio), ``log_or``, ``se``, ``ci``,
        ``information``, ``loglik``, ``n_sets``, ``n_obs``, ``iters``,
        ``converged``.

    References
    ----------
    Breslow, N. E. and Day, N. E. (1980).  Statistical Methods in Cancer
    Research.  Volume I: The Analysis of Case-Control Studies.  IARC
    Scientific Publications No. 32, Lyon, Chapter 7.
    """
    y = [float(t) for t in core.vec(cases)]
    sid = [int(t) for t in core.vec(matching_id)]
    x = [float(t) for t in core.vec(exposure)]
    n = len(y)
    if n == 0:
        raise ValueError("no observations")
    if len(sid) != n or len(x) != n:
        raise ValueError("all inputs must have the same length")
    if any(t not in (0.0, 1.0) for t in y):
        raise ValueError("cases must be coded 0 or 1")
    if controls is not None:
        cc = [float(t) for t in core.vec(controls)]
        if len(cc) != n or any(abs(cc[i] - (1.0 - y[i])) > 1e-12
                               for i in range(n)):
            raise ValueError("controls must be the complement of cases")
    sets = {}
    for i in range(n):
        sets.setdefault(sid[i], []).append(i)
    keys = sorted(sets)
    for k in keys:
        if sum(y[i] for i in sets[k]) != 1.0:
            raise ValueError("every matched set needs exactly one case")
        if len(sets[k]) < 2:
            raise ValueError("every matched set needs at least one control")
    beta = 0.0
    it = 0
    conv = False
    info = 0.0
    for it in range(1, int(max_iter) + 1):
        score = 0.0
        info = 0.0
        for k in keys:
            idx = sets[k]
            mx = max(beta * x[i] for i in idx)
            ex = [math.exp(beta * x[i] - mx) for i in idx]
            s0 = sum(ex)
            s1 = sum(ex[t] * x[idx[t]] for t in range(len(idx)))
            s2 = sum(ex[t] * x[idx[t]] ** 2 for t in range(len(idx)))
            xc = next(x[i] for i in idx if y[i] == 1.0)
            score += xc - s1 / s0
            info += s2 / s0 - (s1 / s0) ** 2
        if info <= 0.0:
            raise ValueError("the conditional information is zero; the "
                             "exposure does not vary within any matched set")
        step = score / info
        beta += step
        if abs(score) < float(tol):
            conv = True
            break
    ll = 0.0
    for k in keys:
        idx = sets[k]
        mx = max(beta * x[i] for i in idx)
        s0 = sum(math.exp(beta * x[i] - mx) for i in idx)
        xc = next(x[i] for i in idx if y[i] == 1.0)
        ll += beta * xc - (mx + math.log(s0))
    se = 1.0 / math.sqrt(info)
    z = core.qnorm(1.0 - (1.0 - float(level)) / 2.0)
    return RichResult(payload={
        "estimate": math.exp(beta), "log_or": beta, "se": se,
        "ci": [math.exp(beta - z * se), math.exp(beta + z * se)],
        "information": info, "loglik": ll, "n_sets": len(keys),
        "n_obs": n, "iters": it, "converged": 1.0 if conv else 0.0,
        "method": "Conditional MLE for matched case-control data"})


def cheatsheet():
    return "matccd: conditional-likelihood odds ratio for matched case-control data"
