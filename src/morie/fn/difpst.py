# morie.fn -- function file (rootcoder007/morie)
"""Differential item functioning: the p-difference and Mantel-Haenszel."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dif_p_diff"]


def dif_p_diff(X, group, focal=1):
    """
    DIF p-difference per item

    Formula: p_focal - p_reference per item

    The raw difference in proportion correct.  It is not by itself
    evidence of item bias, because it confounds impact (a real
    difference in ability) with DIF -- which is why the ETS rules also
    use the Mantel-Haenszel odds ratio on the delta scale,
    -2.35 log(alpha_MH), reported here alongside.  ETS classes: A below
    1, B up to 1.5, C above.

    Parameters
    ----------
    X : array-like
        n x k matrix of 0/1 item responses.
    group : array-like
        Group label per examinee.
    focal : scalar
        Label of the focal group.

    Returns
    -------
    result : dict
        Keys: estimate (largest |p-difference|), p_diff, p_focal,
        p_reference, mh_alpha, mh_delta, ets, flagged, n_focal,
        n_reference, k.

    References
    ----------
    Holland & Wainer (1993), Differential Item Functioning, Erlbaum.
    Dorans & Holland (1993), ibid., ch. 3 (the ETS classification).
    """
    M = core.mat(X)
    n = len(M)
    if n == 0:
        raise ValueError("empty input: X has no rows")
    k = len(M[0])
    g = list(group)
    if len(g) != n:
        raise ValueError("X and group must have the same length")
    for r in M:
        if any(v not in (0.0, 1.0) for v in r):
            raise ValueError("responses must be 0/1")
    fi = [i for i in range(n) if g[i] == focal]
    ri = [i for i in range(n) if g[i] != focal]
    if not fi or not ri:
        raise ValueError("both a focal and a reference group are required")
    total = [sum(M[i]) for i in range(n)]
    pf, pr, pd, alpha, delta, ets, flag = [], [], [], [], [], [], []
    for j in range(k):
        a = sum(M[i][j] for i in fi) / len(fi)
        b = sum(M[i][j] for i in ri) / len(ri)
        pf.append(a)
        pr.append(b)
        pd.append(a - b)
        num = 0.0
        den = 0.0
        for s in range(k + 1):
            f = [i for i in fi if total[i] == s]
            r = [i for i in ri if total[i] == s]
            ns = len(f) + len(r)
            if ns == 0:
                continue
            af = sum(M[i][j] for i in f)
            bf = len(f) - af
            ar = sum(M[i][j] for i in r)
            br = len(r) - ar
            num += ar * bf / ns
            den += af * br / ns
        al = num / den if den > 0.0 else float("nan")
        alpha.append(al)
        dl = -2.35 * math.log(al) if al == al and al > 0.0 else float("nan")
        delta.append(dl)
        ad = abs(dl) if dl == dl else float("nan")
        cls = 0 if (ad == ad and ad < 1.0) else (1 if (ad == ad and ad <= 1.5)
                                                 else 2)
        ets.append(cls)
        flag.append(1 if cls == 2 else 0)
    worst = max(abs(v) for v in pd)
    return RichResult(payload={
        "estimate": worst,
        "p_diff": pd,
        "p_focal": pf,
        "p_reference": pr,
        "mh_alpha": alpha,
        "mh_delta": delta,
        "ets": ets,
        "flagged": flag,
        "n_focal": len(fi),
        "n_reference": len(ri),
        "k": k,
        "method": "DIF p-difference and Mantel-Haenszel delta",
    })


def cheatsheet():
    return "difpst: DIF p-difference per item"


# compact alias per ledger/NAMING.md
difpdiff = dif_p_diff
