# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backdoor-adjusted average treatment effect by stratification."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causbckd", "causal_backdoor_estimate"]


def causbckd(y, x, z):
    """
    Backdoor adjustment formula estimated by stratification on a
    discrete admissible set.

    For an admissible (backdoor) set S, Pearl's adjustment formula is
    P(Y = y | do(X = x)) = sum_s P(Y = y | X = x, S = s) P(S = s)
    (Pearl 2009, Statistics Surveys 3, Eq. 25, Section 3.3.1). Taking
    expectations, the average treatment effect of binary X on Y is

        ATE = sum_z Phat(Z = z) (mean(Y | X = 1, Z = z)
                                 - mean(Y | X = 0, Z = z)),

    the plug-in (stratified) estimator of Eq. 25 with the empirical
    stratum shares Phat(Z = z) = n_z / n. The reported standard error
    treats stratum shares as fixed and the per-stratum arm means as
    independent, giving the elementary delta-method form

        Var = sum_z (n_z / n)^2 (s1_z^2 / n1_z + s0_z^2 / n0_z)

    with s the within-arm sample variances (ddof = 1).

    Parameters
    ----------
    y : array-like
        Outcome, length n.
    x : array-like
        Binary treatment, coded 0/1.
    z : array-like
        Discrete stratum labels of the admissible set (any hashable
        values).

    Returns
    -------
    result : RichResult
        Keys: estimate (ATE), se, strata (per-stratum share, effect,
        counts), n. Strata with an empty treatment or control arm
        raise an error (positivity violation).

    References
    ----------
    Pearl, J. (2009), "Causal inference in statistics: An overview",
    Statistics Surveys 3, 96-146, doi:10.1214/09-SS057, Eq. 25,
    Section 3.3.1 (back-door criterion and adjustment formula).
    Local source: /run/media/rootcoder/WD_BLACK/library/pdf/
    fetched-wave3/pearl-2009-causal-inference-statistics-overview-StatSurveys3.pdf
    """
    y = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    zl = list(z)
    n = len(y)
    if len(xv) != n or len(zl) != n:
        raise ValueError("y, x, z must have equal length")
    for v in xv:
        if float(v) not in (0.0, 1.0):
            raise ValueError("x must be binary 0/1")
    ks = sorted(set(zl), key=lambda v: str(v))
    ate = 0.0
    var = 0.0
    strata = {}
    for k in ks:
        i1 = [i for i in range(n) if zl[i] == k and xv[i] == 1.0]
        i0 = [i for i in range(n) if zl[i] == k and xv[i] == 0.0]
        nz = len(i1) + len(i0)
        if not i1 or not i0:
            raise ValueError(
                "stratum %r has an empty treatment or control arm "
                "(positivity violation)" % (k,))
        y1 = np.asarray([y[i] for i in i1])
        y0 = np.asarray([y[i] for i in i0])
        d = float(np.mean(y1)) - float(np.mean(y0))
        w = nz / float(n)
        ate += w * d
        v1 = float(np.var(y1, ddof=1)) / len(i1) if len(i1) > 1 else 0.0
        v0 = float(np.var(y0, ddof=1)) / len(i0) if len(i0) > 1 else 0.0
        var += w * w * (v1 + v0)
        strata[k] = {"share": w, "effect": d,
                     "n1": len(i1), "n0": len(i0)}
    return RichResult(payload={
        "estimate": ate,
        "se": float(np.sqrt(var)),
        "strata": strata,
        "n": n,
        "method": "Pearl (2009) Eq. 25 backdoor adjustment, stratified plug-in",
    })


causal_backdoor_estimate = causbckd


def cheatsheet():
    return "causbckd(y, x, z) -> backdoor-adjusted ATE by stratification on an admissible set."
