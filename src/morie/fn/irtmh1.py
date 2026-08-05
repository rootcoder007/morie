# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Mantel-Haenszel differential item functioning.

Holland and Thayer (1988), "Differential item performance and the
Mantel-Haenszel procedure", in Wainer and Braun (eds), *Test
Validity*, Lawrence Erlbaum, pp. 129-145.  Examinees are stratified on
the matching total score; each stratum m gives a 2 x 2 table

               correct   wrong    total
    reference    A_m      B_m      n_Rm
    focal        C_m      D_m      n_Fm

and the common odds ratio is

    alpha_MH = sum_m A_m D_m / n_m  /  sum_m B_m C_m / n_m.

The MH chi-square uses the hypergeometric mean and variance of A_m
with Holland and Thayer's continuity correction, and the ETS delta
scale is Delta = -2.35 ln(alpha_MH); the A/B/C classification follows
the ETS rule (|Delta| < 1 or not significant -> A; |Delta| >= 1.5 and
significant -> C; otherwise B).
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["dif_mantel_haenszel"]


def _one_item(x, g, s, strata):
    num = 0.0
    den = 0.0
    ea = 0.0
    va = 0.0
    obs = 0.0
    for lev in strata:
        idx = [i for i in range(len(x)) if s[i] == lev]
        if not idx:
            continue
        A = sum(1.0 for i in idx if g[i] == 0 and x[i] == 1)
        B = sum(1.0 for i in idx if g[i] == 0 and x[i] == 0)
        C = sum(1.0 for i in idx if g[i] == 1 and x[i] == 1)
        D = sum(1.0 for i in idx if g[i] == 1 and x[i] == 0)
        n = A + B + C + D
        if n <= 1:
            continue
        nR = A + B
        nF = C + D
        n1 = A + C
        n0 = B + D
        if nR == 0 or nF == 0 or n1 == 0 or n0 == 0:
            continue
        num += A * D / n
        den += B * C / n
        obs += A
        ea += nR * n1 / n
        va += nR * nF * n1 * n0 / (n * n * (n - 1.0))
    if den <= 0 or num <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    orat = num / den
    if va <= 0:
        return orat, float("nan"), float("nan"), -2.35 * math.log(orat)
    chi = (abs(obs - ea) - 0.5) ** 2 / va
    p = 2.0 * (1.0 - core.pnorm(math.sqrt(chi)))
    return orat, chi, p, -2.35 * math.log(orat)


def dif_mantel_haenszel(X, group, total_score, alpha=0.05):
    """MH odds ratio, chi-square, ETS delta and A/B/C class per item.

    Parameters
    ----------
    X : n x J matrix of 0/1 item responses.
    group : length-n vector, 0 for the reference group, 1 for the focal.
    total_score : length-n matching variable used to form the strata.
    alpha : significance level for the MH chi-square.
    """
    M = core.mat(X)
    n = len(M)
    if n == 0:
        raise ValueError("dif_mantel_haenszel: X is empty")
    J = len(M[0])
    g = [int(v) for v in core.vec(group)]
    s = core.vec(total_score)
    if len(g) != n or len(s) != n:
        raise ValueError("dif_mantel_haenszel: group and total_score must match X")
    for v in g:
        if v not in (0, 1):
            raise ValueError("dif_mantel_haenszel: group must be coded 0/1")
    strata = sorted(set(s))
    ors = []
    chis = []
    ps = []
    deltas = []
    cls = []
    flags = []
    for j in range(J):
        col = [int(M[i][j]) for i in range(n)]
        for v in col:
            if v not in (0, 1):
                raise ValueError("dif_mantel_haenszel: responses must be 0/1")
        o, c, p, d = _one_item(col, g, s, strata)
        sig = (p == p) and p < alpha
        if not (d == d) or (abs(d) < 1.0) or not sig:
            k = "A"
        elif abs(d) >= 1.5 and sig:
            k = "C"
        else:
            k = "B"
        ors.append(o)
        chis.append(c)
        ps.append(p)
        deltas.append(d)
        cls.append(k)
        flags.append(0 if k == "A" else 1)
    fin = [v for v in deltas if v == v]
    return RichResult(
        title="Mantel-Haenszel DIF",
        summary_lines=[("items", J), ("strata", len(strata))],
        payload={
            "estimate": max((abs(v) for v in fin), default=float("nan")),
            "odds_ratio": ors,
            "chisq": chis,
            "p_value": ps,
            "delta": deltas,
            "ets_class": cls,
            "flagged": flags,
            "n": n,
            "method": "MH common odds ratio with ETS delta, Holland & Thayer (1988)",
        },
    )


def cheatsheet():
    return "irtmh1: Mantel-Haenszel DIF detection"
