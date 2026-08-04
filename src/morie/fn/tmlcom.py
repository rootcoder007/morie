# morie.fn -- function file (rootcoder007/morie)
"""Targeted estimate for a compositional outcome."""

import math

from . import _tail1core as C
from . import _b1tmle as T

from ._richresult import RichResult

__all__ = ["comptml", "tmle_compositional"]


def comptml(Yc, A, Q1, Q0, g1W, gbound=0.025, level=0.95):
    """Treatment effect on a composition, estimated in clr coordinates.

    A treatment effect on a composition cannot be read off the parts:
    the parts sum to a constant, so pushing one up MUST push the
    others down and every "effect" is confounded by the closure.  The
    effect is therefore estimated on the clr coordinates, where the
    constraint has been removed -- and the returned effects sum to
    zero for exactly the same reason the clr does.

    Each coordinate is targeted separately with the shared propensity,
    since one fluctuation cannot solve D coordinate score equations at
    once.  The outcome is mapped to [0, 1] by the standard affine
    transform before the logistic fluctuation and mapped back, which
    is what lets a bounded-continuous outcome use the same machinery.

    Formula: Z = clr(Y);  for each coordinate j, target
             psi_j = E[Z_j(1)] - E[Z_j(0)];
             sum_j psi_j = 0 by construction;
             the perturbation on the simplex is C(exp(psi))

    Parameters
    ----------
    Yc : array-like, shape (n, D)
        Compositional outcome, one composition per row, strictly
        positive.
    A : array-like
        Binary treatment.
    Q1, Q0 : array-like, shape (n, D)
        Initial predictions of E[clr(Y)_j | A = 1, W] and A = 0.
    g1W : array-like
        Initial propensity.
    gbound : float
        Propensity truncation.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``effect`` (per clr coordinate), ``se``, ``ci_lower``,
        ``ci_upper``, ``sum_effect`` (zero by construction),
        ``perturbation`` (the effect as a composition), ``n``, ``D``.

    References
    ----------
    Targeting machinery verified against the reference implementation
    in the CRAN package ``tmle`` 2.1.1 (Gruber & van der Laan); the
    bounded-continuous-outcome transform is Gruber & van der Laan
    (2010), International Journal of Biostatistics 6(1), Article 26.
    The compositional geometry -- clr coordinates and the
    interpretation of a difference there as a perturbation on the
    simplex -- is Aitchison (1986), The Statistical Analysis of
    Compositional Data, Chapters 2 and 4, matching the sibling modules
    ``aitclr`` and ``aitprt`` in this package.  No source combining the
    two was found; the combination is documented here as this
    package's own.
    """
    Yc = C.mat(Yc)
    n = len(Yc)
    if n < 2:
        raise ValueError("at least two observations are required")
    D = len(Yc[0])
    if D < 2:
        raise ValueError("a composition needs at least two parts")
    if any(len(r) != D for r in Yc):
        raise ValueError("every composition must have the same length")
    for r in Yc:
        if any(v <= 0 for v in r):
            raise ValueError("compositions must be strictly positive")
    A = C.vec(A)
    Q1 = C.mat(Q1)
    Q0 = C.mat(Q0)
    g1W = C.vec(g1W)
    if len(A) != n or len(g1W) != n or len(Q1) != n or len(Q0) != n:
        raise ValueError("every argument must have one entry per observation")
    if any(len(r) != D for r in Q1) or any(len(r) != D for r in Q0):
        raise ValueError("Q1 and Q0 must have one column per part")
    if any(v not in (0.0, 1.0) for v in A):
        raise ValueError("A must be binary 0/1")
    Z = []
    for r in Yc:
        L = [math.log(v) for v in r]
        g = sum(L) / D
        Z.append([v - g for v in L])
    eff = []
    ses = []
    lo = []
    hi = []
    z = C.qnorm((1.0 + float(level)) / 2.0)
    for j in range(D):
        col = [Z[i][j] for i in range(n)]
        pred = [Q1[i][j] for i in range(n)] + [Q0[i][j] for i in range(n)]
        a = min(min(col), min(pred))
        b = max(max(col), max(pred))
        rng = b - a
        if rng <= 0:
            raise ValueError("a clr coordinate is constant; no effect to target")
        Ys = [(v - a) / rng for v in col]
        q1 = [T.bound((Q1[i][j] - a) / rng, 1e-6, 1 - 1e-6) for i in range(n)]
        q0 = [T.bound((Q0[i][j] - a) / rng, 1e-6, 1 - 1e-6) for i in range(n)]
        qa = [q1[i] if A[i] == 1.0 else q0[i] for i in range(n)]
        fit = T.target(Ys, A, qa, q1, q0, g1W, gbound)
        mu1, mu0, ic1, ic0 = T.curves(Ys, A, fit)
        ic = [(ic1[i] - ic0[i]) * rng for i in range(n)]
        e = (mu1 - mu0) * rng
        s = math.sqrt(C.var(ic, 1) / n)
        eff.append(e)
        ses.append(s)
        lo.append(e - z * s)
        hi.append(e + z * s)
    # Recentre so the effects sum to zero, as a clr difference must.
    mean_e = sum(eff) / D
    eff = [v - mean_e for v in eff]
    ex = [math.exp(v) for v in eff]
    s = sum(ex)
    return RichResult(payload={
        "effect": eff, "se": ses, "ci_lower": [v - mean_e for v in lo],
        "ci_upper": [v - mean_e for v in hi], "sum_effect": sum(eff),
        "perturbation": [v / s for v in ex], "n": float(n),
        "D": float(D),
        "method": "TMLE on clr coordinates of a compositional outcome"})


tmle_compositional = comptml


def cheatsheet():
    return "tmlcom: target each clr coordinate; effects sum to zero, map to a perturbation"
