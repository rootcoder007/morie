# morie.fn -- function file (rootcoder007/morie)
"""Effect modification on the additive and the multiplicative scale.

Source opened: Knol, M. J. and VanderWeele, T. J. (2012).
Recommendations for presenting analyses of effect modification and
interaction.  *International Journal of Epidemiology* 41(2), 514-520,
doi:10.1093/ije/dyr218.  The paper's recommendation is that a single
reference cell be used for all four exposure-by-modifier strata, and
that BOTH scales be reported, because a positive interaction on one
scale is compatible with none or a negative one on the other.

With A the exposure, V the modifier and p_av the risk in cell (A = a,
V = v), the reported quantities are

    RR_av  = p_av / p_00                   (single reference cell)
    RERI   = RR_11 - RR_10 - RR_01 + 1     additive scale
    mult   = RR_11 / (RR_10 RR_01)         multiplicative scale
    RD_int = p_11 - p_10 - p_01 + p_00     additive, on the risk scale
    AP     = RERI / RR_11                  attributable proportion

The paper's own worked example is RERI = 2.07 - 1.55 - 1.16 + 1 = 0.36,
which is the printed arithmetic this module is anchored on.  Note the
sign convention: RERI is zero under exact additivity of risks, and
mult is one under exact multiplicativity; a table can have RERI > 0
with mult < 1, which is the whole point of reporting both.

When covariates H are supplied the four cell risks are the fitted
values of a least-squares regression of the outcome on the four cell
indicators and the mean-centred covariates, so each coefficient IS the
covariate-adjusted risk of its cell at the covariate mean.  A linear
probability model is used because it keeps the additive scale exactly
additive; with H omitted the coefficients reduce to the crude cell
proportions.
"""

from __future__ import annotations

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["effect_modification"]


def effect_modification(y, A, V, H=None):
    """Interaction of A and V on both scales, single reference cell.

    Parameters
    ----------
    y : array-like
        Binary outcome, 0/1.
    A : array-like
        Binary exposure.
    V : array-like
        Binary effect modifier.
    H : 2-D array-like, optional
        Covariates to adjust for; mean-centred internally.

    Returns
    -------
    result : dict
        Keys: estimate (RERI), p00, p10, p01, p11, rr10, rr01, rr11,
        reri, mult, rd_int, ap, n00, n10, n01, n11, n.

    References
    ----------
    Knol & VanderWeele (2012), Int. J. Epidemiol. 41(2):514-520,
    doi:10.1093/ije/dyr218.
    """
    yv = core.vec(y)
    av = core.vec(A)
    vv = core.vec(V)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(av) != n or len(vv) != n:
        raise ValueError("y, A and V must have the same length")
    for v in av:
        if v != 0.0 and v != 1.0:
            raise ValueError("A must be binary 0/1")
    for v in vv:
        if v != 0.0 and v != 1.0:
            raise ValueError("V must be binary 0/1")
    cells = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    idx = [[i for i in range(n) if av[i] == a and vv[i] == v]
           for (a, v) in cells]
    cnt = [float(len(z)) for z in idx]
    for j in range(4):
        if cnt[j] == 0.0:
            raise ValueError("every A x V cell must be non-empty")
    if H is None:
        p = [core.mean([yv[i] for i in z]) for z in idx]
    else:
        Hr = core.mat(H)
        if len(Hr) != n:
            raise ValueError("H must have one row per observation")
        q = len(Hr[0])
        cm = [core.mean([Hr[i][j] for i in range(n)]) for j in range(q)]
        Z = []
        for i in range(n):
            row = [0.0, 0.0, 0.0, 0.0]
            for k in range(4):
                if av[i] == cells[k][0] and vv[i] == cells[k][1]:
                    row[k] = 1.0
            Z.append(row + [Hr[i][j] - cm[j] for j in range(q)])
        beta = core.lstsq(Z, yv)
        p = [beta[j] for j in range(4)]
    p00, p10, p01, p11 = p
    if p00 <= 0.0:
        raise ValueError("the reference cell risk must be strictly positive")
    rr10, rr01, rr11 = p10 / p00, p01 / p00, p11 / p00
    reri = rr11 - rr10 - rr01 + 1.0
    mult = rr11 / (rr10 * rr01) if (rr10 > 0.0 and rr01 > 0.0) else float("nan")
    return RichResult(
        title="Effect modification, both scales",
        summary_lines=[("RERI", reri), ("multiplicative", mult)],
        payload={
            "estimate": reri,
            "p00": p00, "p10": p10, "p01": p01, "p11": p11,
            "rr10": rr10, "rr01": rr01, "rr11": rr11,
            "reri": reri,
            "mult": mult,
            "rd_int": p11 - p10 - p01 + p00,
            "ap": reri / rr11 if rr11 != 0.0 else float("nan"),
            "n00": cnt[0], "n10": cnt[1], "n01": cnt[2], "n11": cnt[3],
            "n": n,
            "method": "Effect modification on the additive vs multiplicative scale",
        },
    )


def cheatsheet():
    return "effmod: Effect modification on the additive vs multiplicative scale"
