# morie.fn -- slice k04 (rootcoder007/morie)
"""SIBTEST differential item functioning (Shealy and Stout 1993).

Source FETCHED (reference implementation): ``SIBTEST`` in the CRAN
package ``mirt`` (Chalmers, mirt 1.46.1, file ``R/SIBTEST.R``), which
implements Shealy, R. and Stout, W. (1993), "A model-based standardization
approach that separates true bias/DIF from group ability differences and
detects test bias/DTF as well as item bias/DIF", *Psychometrika* 58,
159-194.  (``difR::sibTest`` is a thin wrapper that delegates to it.)
The 1993 paper is paywalled here; the package source states the
estimator explicitly.  Examinees are grouped on the matching score k:

    pstar_k       = n_k / sum_k n_k          (weights from both groups)
    Ybar_R,k      = mean suspect-item score, reference group, at k
    Ybar_F,k      = mean suspect-item score, focal group, at k

    beta_hat = sum_k pstar_k ( Ystar_R,k - Ystar_F,k )

    sigma    = sqrt( sum_k pstar_k^2 ( s2_F,k / n_F,k + s2_R,k / n_R,k ) )

    X2 = (beta_hat / sigma)^2       ~ chi^2(1) under H0: beta = 0

with s2 the within-cell sample variance.  A score level contributes only
when both groups have examinees there AND both within-cell variances are
non-zero -- ``mirt`` drops the rest and renormalises ``pstar`` over what
is left, and so does this, because a level with no variance contributes
nothing to sigma and would otherwise make it an underestimate.

``Ystar`` is ``Ybar`` unless ``correction=True``, which applies the
Shealy-Stout true-score regression correction as ``mirt`` writes it::

    M_g       = (Ybar_g,k+1 - Ybar_g,k-1) / (V_g,k+1 - V_g,k-1)
    Ystar_g,k = Ybar_g,k + M_g (V_k - V_g,k)

with V the mean matching score in the cell, pooled (V_k) or per group
(V_g,k).  The default here is ``correction=False``, which is exactly the
estimator this row's docstring specifies; turn it on for SIBTEST as the
1993 paper defines it.  The correction is skipped at the two end levels,
where k-1 or k+1 does not exist.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["dif_sibtest"]


def _cell(vals):
    n = len(vals)
    if n == 0:
        return 0, float("nan"), float("nan")
    m = sum(vals) / n
    if n < 2:
        return n, m, 0.0
    v = sum((a - m) ** 2 for a in vals) / (n - 1)
    return n, m, v


def dif_sibtest(y, group, studied=None, matching=None, correction=False):
    """SIBTEST beta-hat and its chi-square for one suspect item.

    Parameters
    ----------
    y : array-like
        Score on the suspect (studied) item, one per examinee.  Binary or
        polytomous.
    group : array-like
        Group membership; exactly two distinct values.  The first value
        encountered is the reference group.
    studied : array-like, optional
        Alias for ``y``, accepted for the stub's original argument order
        ``(y, group, studied, matching)``.  When given it overrides ``y``
        as the suspect-item score and ``y`` is ignored.
    matching : array-like
        The matching (valid subtest) score, one per examinee.  Examinees
        are grouped on its distinct values.  Required.
    correction : bool, default False
        Apply the Shealy-Stout true-score regression correction.

    Returns
    -------
    RichResult
        keys: ``beta``, ``sigma``, ``statistic`` (X2), ``p_value``,
        ``df``, ``n_levels`` (levels that contributed), ``levels``,
        ``pstar``, ``correction``, ``n``, ``method``.
    """
    suspect = y if studied is None else studied
    s = np.asarray(suspect, dtype=float).ravel()
    n = int(s.size)
    g = list(group)
    if len(g) != n:
        raise ValueError("group must be the same length as the item score")
    if matching is None:
        raise ValueError("matching score is required")
    mt = list(matching)
    if len(mt) != n:
        raise ValueError("matching must be the same length as the item score")

    levels = []
    for a in g:
        if a not in levels:
            levels.append(a)
    if len(levels) != 2:
        raise ValueError(f"group must have exactly 2 distinct values; saw {len(levels)}")
    ref = levels[0]

    keys = sorted(set(mt))
    tab, ybar_r, ybar_f, s2r, s2f, nr, nf, vr, vf, vk = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for k in keys:
        idx = [i for i in range(n) if mt[i] == k]
        ir = [i for i in idx if g[i] == ref]
        i_f = [i for i in idx if g[i] != ref]
        tab[k] = len(idx)
        nr[k], ybar_r[k], s2r[k] = _cell([float(s[i]) for i in ir])
        nf[k], ybar_f[k], s2f[k] = _cell([float(s[i]) for i in i_f])
        vr[k] = sum(float(mt[i]) for i in ir) / len(ir) if ir else float("nan")
        vf[k] = sum(float(mt[i]) for i in i_f) / len(i_f) if i_f else float("nan")
        vk[k] = sum(float(mt[i]) for i in idx) / len(idx)

    keep = [
        k
        for k in keys
        if nr[k] > 0 and nf[k] > 0 and s2r[k] == s2r[k] and s2f[k] == s2f[k]
        and s2r[k] > 0.0 and s2f[k] > 0.0
    ]
    if not keep:
        raise ValueError("no matching level has both groups present with non-zero within-cell variance")

    tot = float(sum(tab[k] for k in keep))
    pstar = {k: tab[k] / tot for k in keep}

    ystar_r, ystar_f = {}, {}
    for k in keep:
        if correction:
            j = keys.index(k)
            if 0 < j < len(keys) - 1:
                kp, km = keys[j + 1], keys[j - 1]
                dr = vr[kp] - vr[km]
                df_ = vf[kp] - vf[km]
                mr = (ybar_r[kp] - ybar_r[km]) / dr if dr not in (0.0,) and dr == dr else 0.0
                mf = (ybar_f[kp] - ybar_f[km]) / df_ if df_ not in (0.0,) and df_ == df_ else 0.0
                ystar_r[k] = ybar_r[k] + mr * (vk[k] - vr[k])
                ystar_f[k] = ybar_f[k] + mf * (vk[k] - vf[k])
                continue
        ystar_r[k] = ybar_r[k]
        ystar_f[k] = ybar_f[k]

    beta = sum(pstar[k] * (ystar_r[k] - ystar_f[k]) for k in keep)
    var = sum(pstar[k] ** 2 * (s2f[k] / nf[k] + s2r[k] / nr[k]) for k in keep)
    sigma = math.sqrt(var) if var > 0.0 else float("nan")
    stat = (beta / sigma) ** 2 if sigma == sigma and sigma > 0.0 else float("nan")
    p = float(stats.chi2.sf(stat, 1)) if stat == stat else float("nan")
    return RichResult(
        payload={
            "beta": float(beta),
            "sigma": float(sigma),
            "statistic": float(stat),
            "p_value": p,
            "df": 1,
            "n_levels": len(keep),
            "levels": np.array([float(k) for k in keep], dtype=float),
            "pstar": np.array([pstar[k] for k in keep], dtype=float),
            "correction": bool(correction),
            "n": n,
            "method": "SIBTEST DIF (Shealy and Stout 1993; mirt::SIBTEST)",
        }
    )


def cheatsheet():
    return "difsib: SIBTEST differential item functioning"
