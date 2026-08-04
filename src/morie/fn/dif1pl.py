# morie.fn -- slice k04 (rootcoder007/morie)
"""Mantel-Haenszel differential item functioning (DIF) statistic.

Source: Holland, P. W. and Thayer, D. T. (1988), "Differential item
performance and the Mantel-Haenszel procedure", in Wainer and Braun
(eds), *Test Validity*, 129-145, applying Mantel and Haenszel (1959),
*Journal of the National Cancer Institute* 22, 719-748.  Neither
chapter was obtainable here; the procedure is quoted in its standard
published form, which is unambiguous and is exactly the stratified 2x2
statistic base R implements in ``stats::mantelhaen.test`` (used as the
independent anchor for this function).

Examinees are stratified by matching score k.  In stratum k the item
gives the 2 x 2 table

                 right      wrong     total
    reference     A_k        B_k      n1_k
    focal         C_k        D_k      n2_k
    total         m1_k       m0_k      T_k

with, under the null of no DIF,

    E[A_k]   = n1_k m1_k / T_k
    Var[A_k] = n1_k n2_k m1_k m0_k / ( T_k^2 (T_k - 1) )

    chi2_MH = ( |sum_k A_k - sum_k E[A_k]| - 0.5 )^2 / sum_k Var[A_k]

on one degree of freedom, and the constant-odds-ratio estimator

    alpha_MH = sum_k (A_k D_k / T_k) / sum_k (B_k C_k / T_k),

from which ETS reports the delta scale  -2.35 log(alpha_MH).  Strata
with T_k < 2, or with no variation in the item or in group membership,
contribute nothing and are dropped, as they must be: their variance
term is zero or undefined.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["dif_mantel_haenszel"]


def dif_mantel_haenszel(y, group, item=None, correct=True):
    """Mantel-Haenszel DIF chi-square for one item.

    Parameters
    ----------
    y : array-like of {0, 1}
        Response to the studied item, one per examinee.
    group : array-like
        Group membership.  The first distinct value encountered is the
        reference group; everything else is the focal group.  Pass two
        distinct values.
    item : array-like, optional
        Matching variable (normally the total score), one per examinee.
        Examinees are stratified on its distinct values.  If omitted,
        every examinee falls in a single stratum, which reduces the
        statistic to an ordinary 2 x 2 chi-square.
    correct : bool, default True
        Apply the 0.5 continuity correction, as Holland and Thayer do.

    Returns
    -------
    RichResult
        keys: ``statistic``, ``p_value``, ``df``, ``alpha_MH``,
        ``delta_MH``, ``sum_A``, ``sum_E``, ``sum_V``, ``n_strata``,
        ``n_used``, ``n``, ``method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)
    g = list(group)
    if len(g) != n:
        raise ValueError("group must be the same length as y")
    if not bool(np.all((y == 0.0) | (y == 1.0))):
        raise ValueError("y must be coded 0/1")
    levels = []
    for a in g:
        if a not in levels:
            levels.append(a)
    if len(levels) != 2:
        raise ValueError(f"group must have exactly 2 distinct values; saw {len(levels)}")
    ref = levels[0]
    is_ref = [a == ref for a in g]

    if item is None:
        strata = [0] * n
    else:
        strata = list(item)
        if len(strata) != n:
            raise ValueError("item must be the same length as y")
    keys = []
    for s in strata:
        if s not in keys:
            keys.append(s)

    sum_a = 0.0
    sum_e = 0.0
    sum_v = 0.0
    num = 0.0
    den = 0.0
    used = 0
    n_used = 0
    for kk in keys:
        idx = [i for i in range(n) if strata[i] == kk]
        T = float(len(idx))
        if T < 2.0:
            continue
        A = float(sum(1 for i in idx if is_ref[i] and y[i] == 1.0))
        B = float(sum(1 for i in idx if is_ref[i] and y[i] == 0.0))
        C = float(sum(1 for i in idx if not is_ref[i] and y[i] == 1.0))
        D = float(sum(1 for i in idx if not is_ref[i] and y[i] == 0.0))
        n1 = A + B
        n2 = C + D
        m1 = A + C
        m0 = B + D
        if n1 == 0.0 or n2 == 0.0 or m1 == 0.0 or m0 == 0.0:
            continue  # no information: variance term is 0
        sum_a += A
        sum_e += n1 * m1 / T
        sum_v += n1 * n2 * m1 * m0 / (T * T * (T - 1.0))
        num += A * D / T
        den += B * C / T
        used += 1
        n_used += len(idx)

    if used == 0 or sum_v <= 0.0:
        raise ValueError("no stratum carries information about DIF")

    d = abs(sum_a - sum_e)
    if correct:
        d = max(0.0, d - 0.5)
    stat = d * d / sum_v
    alpha_mh = num / den if den > 0.0 else float("inf")
    delta_mh = -2.35 * math.log(alpha_mh) if 0.0 < alpha_mh < float("inf") else float("nan")
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(stats.chi2.sf(stat, 1)),
            "df": 1,
            "alpha_MH": float(alpha_mh),
            "delta_MH": float(delta_mh),
            "sum_A": float(sum_a),
            "sum_E": float(sum_e),
            "sum_V": float(sum_v),
            "n_strata": used,
            "n_used": n_used,
            "n": n,
            "method": "Mantel-Haenszel DIF chi-square (Holland and Thayer 1988; Mantel and Haenszel 1959)",
        }
    )


def cheatsheet():
    return "dif1pl: Mantel-Haenszel DIF chi-square"
