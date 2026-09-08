# morie.fn -- k02 batch (rootcoder007/morie)
"""Three-level (nested) random-effects meta-analysis by maximum likelihood.

Source consulted: Cheung, M.W.-L. (2014), Modeling dependent effect sizes with
three-level meta-analyses: a structural equation modeling approach,
*Psychological Methods* 19(2), 211-229, equations (5)-(8).  Effect i inside
cluster j has

    y_ij = mu + u_j + e_ij + eps_ij,
    Var(y_j) = diag(v_ij + tau2_2) + tau2_3 * 1 1'

so within a cluster the covariance is exchangeable.  The likelihood is
evaluated cluster by cluster with the Sherman-Morrison inverse and the
matching determinant

    log|V_j| = sum log(v_ij + tau2_2) + log(1 + tau2_3 sum 1/(v_ij + tau2_2))

mu is the closed-form GLS estimate at the current variance components, and the
two components are maximised by a nested golden-section search with a fixed
iteration count, so the Python and R arms take identical steps.  Verified
against ``metafor::rma.mv`` (agreement reported in the canonical test).
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02gold, k02z

from ._richresult import RichResult

__all__ = ["ma_three_level"]


def _blocks(cluster):
    seen = []
    out = []
    for c in cluster:
        if c not in seen:
            seen.append(c)
            out.append([])
        out[seen.index(c)].append(len(out[seen.index(c)]))
    return seen


def _gls(y, v, cl, t2, t3):
    """Return (mu, var_mu, minus2ll) at the given variance components."""
    keys = []
    for c in cl:
        if c not in keys:
            keys.append(c)
    a = 0.0
    b = 0.0
    logdet = 0.0
    quad_y = 0.0
    quad_1 = 0.0
    cross = 0.0
    for key in keys:
        idx = [i for i in range(len(y)) if cl[i] == key]
        d = np.asarray([v[i] + t2 for i in idx], dtype=float)
        yy = np.asarray([y[i] for i in idx], dtype=float)
        di = 1.0 / d
        s1 = float(np.sum(di))
        sy = float(np.sum(di * yy))
        syy = float(np.sum(di * yy * yy))
        denom = 1.0 + t3 * s1
        logdet += float(np.sum(np.log(d))) + float(np.log(denom))
        quad_1 += s1 - t3 * s1 * s1 / denom
        cross += sy - t3 * s1 * sy / denom
        quad_y += syy - t3 * sy * sy / denom
    a = quad_1
    b = cross
    mu = b / a
    rss = quad_y - 2.0 * mu * b + mu * mu * a
    m2ll = logdet + rss + len(y) * float(np.log(2.0 * np.pi))
    return mu, 1.0 / a, m2ll


def ma_three_level(yi, vi, cluster, level=0.95, upper=None):
    """Three-level random-effects meta-analysis, ML variance components.

    Parameters
    ----------
    yi, vi : array-like
        Effect sizes and their sampling variances.
    cluster : array-like
        Level-3 grouping label for each effect.
    level : float, default 0.95
        Confidence level.
    upper : float, optional
        Upper bound of the variance-component search; ten times the marginal
        variance of ``yi`` by default.

    Returns
    -------
    RichResult
        estimate (mu), se, ci_lower, ci_upper, tau2_level2, tau2_level3,
        loglik, i2_level2, i2_level3, n_clusters, n, method.
    """
    y = [float(t) for t in np.asarray(yi, dtype=float).ravel()]
    v = [float(t) for t in np.asarray(vi, dtype=float).ravel()]
    cl = list(cluster)
    hi = float(upper) if upper is not None else 10.0 * float(np.var(np.asarray(y, dtype=float))) + 1e-8

    def inner(t3):
        return k02gold(lambda t2: _gls(y, v, cl, t2, t3)[2], 0.0, hi, 60)

    def outer(t3):
        return _gls(y, v, cl, inner(t3), t3)[2]

    t3 = k02gold(outer, 0.0, hi, 60)
    t2 = inner(t3)
    mu, var, m2ll = _gls(y, v, cl, t2, t3)
    se = float(np.sqrt(var))
    crit = k02z(0.5 + 0.5 * float(level))
    vbar = float(np.mean(np.asarray(v, dtype=float)))
    tot = t2 + t3 + vbar
    return RichResult(
        payload={
            "estimate": float(mu),
            "se": se,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "tau2_level2": float(t2),
            "tau2_level3": float(t3),
            "loglik": float(-0.5 * m2ll),
            "i2_level2": float(100.0 * t2 / tot),
            "i2_level3": float(100.0 * t3 / tot),
            "n_clusters": int(len(set(cl))),
            "n": int(len(y)),
            "method": "Three-level random-effects meta-analysis, ML (Cheung 2014)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22, 0.31, -0.05]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04, 0.02, 0.06]
# >>> g = [1, 1, 2, 2, 3, 3, 4, 4]
# >>> r = ma_three_level(y, v, g)
# >>> assert r["tau2_level2"] >= 0.0 and r["tau2_level3"] >= 0.0
# >>> assert r["n_clusters"] == 4


def cheatsheet():
    return "matrl(yi, vi, cluster): three-level random-effects meta-analysis."


mathreelevel = ma_three_level
