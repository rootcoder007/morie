# morie.fn -- function file (rootcoder007/morie)
"""Targeted estimates for a categorical treatment."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["tmlecat", "tmle_categorical_outcome"]


def tmlecat(Y, A, Q, G, ref=1, gbound=0.025, level=0.95):
    """Target E[Y_a] at every level of a categorical treatment.

    With more than two levels each level gets its OWN clever covariate
    1{A = a}/g_a(W) and its own fluctuation, because a single logistic
    update cannot solve L score equations at once.  Doing them
    separately is what keeps every psi_a efficient rather than only
    the contrast that happened to be fitted.

    Positivity is the binding assumption here and it gets harder with
    every level added: ``min_g`` is returned per level so a level with
    almost no support is visible instead of silently producing a huge
    weight.

    ``A`` holds ONE-BASED level labels 1..L, matching the R arm.

    Formula: for each level a, fluctuate logit Q(a,W) on
             H_a = 1{A = a}/g_a(W); psi_a = mean(Q*(a,W));
             IC_a = 1{A=a}/g_a (Y - Q*(A,W)) + Q*(a,W) - psi_a

    Parameters
    ----------
    Y : array-like
        Outcome in [0, 1].
    A : array-like of int
        One-based treatment level of each observation.
    Q : array-like, shape (n, L)
        Initial predictions of E[Y | A = a, W], column a per level.
    G : array-like, shape (n, L)
        Initial P(A = a | W), rows summing to 1.
    ref : int
        One-based reference level for the contrasts.
    gbound : float
        Truncation applied to each g_a.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``psi`` (per level), ``se``, ``contrast`` (psi_a - psi_ref),
        ``contrast_se``, ``ci_lower``, ``ci_upper``, ``min_g``,
        ``ref``, ``n``, ``L``.

    References
    ----------
    Verified against the reference implementation in the CRAN package
    ``tmle`` 2.1.1 (Gruber & van der Laan): the binary case reduces to
    its ``glm(Ystar ~ -1 + offset(qlogis(QAW)) + H0W + H1W)`` with
    H1W = A/g1W, and the multi-level extension applies the same
    fluctuation per level, as in van der Laan & Rose (2011), Targeted
    Learning, Chapter 4.  The row's own citation is that book.
    """
    Y = C.vec(Y)
    n = len(Y)
    A = [int(v) for v in C.vec(A)]
    Qm = C.mat(Q)
    Gm = C.mat(G)
    if len(A) != n or len(Qm) != n or len(Gm) != n:
        raise ValueError("every argument must have one entry per observation")
    L = len(Qm[0])
    if L < 2:
        raise ValueError("at least two treatment levels are required")
    if any(len(r) != L for r in Qm) or any(len(r) != L for r in Gm):
        raise ValueError("Q and G must have one column per level")
    if any(not 1 <= v <= L for v in A):
        raise ValueError("A must hold one-based level labels in 1..L")
    ref = int(ref)
    if not 1 <= ref <= L:
        raise ValueError("ref must be a level in 1..L")
    if any(v < 0.0 or v > 1.0 for v in Y):
        raise ValueError("Y must lie in [0, 1]")
    psi = []
    ses = []
    ics = []
    mg = []
    for a in range(1, L + 1):
        ind = [1.0 if A[i] == a else 0.0 for i in range(n)]
        ga = [T.bound(Gm[i][a - 1], gbound, 1.0 - gbound) for i in range(n)]
        mg.append(min(ga))
        H = [ind[i] / ga[i] for i in range(n)]
        QAW = [Qm[i][A[i] - 1] for i in range(n)]
        off = [T.logit(v) for v in QAW]
        e = 0.0
        for _ in range(100):
            gr = 0.0
            hs = 1e-10
            for i in range(n):
                mu = T.expit(off[i] + e * H[i])
                gr += H[i] * (Y[i] - mu)
                hs += H[i] * H[i] * mu * (1.0 - mu)
            st = gr / hs
            e += st
            if abs(st) < 1e-12:
                break
        QAs = [T.expit(off[i] + e * H[i]) for i in range(n)]
        Qas = [T.expit(T.logit(Qm[i][a - 1]) + e / ga[i]) for i in range(n)]
        p = sum(Qas) / n
        ic = [H[i] * (Y[i] - QAs[i]) + Qas[i] - p for i in range(n)]
        psi.append(p)
        ics.append(ic)
        ses.append(math.sqrt(C.var(ic, 1) / n))
    z = C.qnorm((1.0 + float(level)) / 2.0)
    con = []
    cse = []
    lo = []
    hi = []
    for a in range(L):
        d = psi[a] - psi[ref - 1]
        icd = [ics[a][i] - ics[ref - 1][i] for i in range(n)]
        s = math.sqrt(C.var(icd, 1) / n)
        con.append(d)
        cse.append(s)
        lo.append(d - z * s)
        hi.append(d + z * s)
    return RichResult(payload={
        "psi": psi, "se": ses, "contrast": con, "contrast_se": cse,
        "ci_lower": lo, "ci_upper": hi, "min_g": mg, "ref": float(ref),
        "n": float(n), "L": float(L),
        "method": "TMLE for a categorical treatment, one fluctuation per level"})


tmle_categorical_outcome = tmlecat


def cheatsheet():
    return "tmlcat: one clever covariate 1{A=a}/g_a and one fluctuation per level"
