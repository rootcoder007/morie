# morie.fn -- function file (rootcoder007/morie)
"""Manski-Pepper MTR-MTS bounds (the returns-to-schooling application)."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bndapp", "bound_application"]


def bndapp(y, z, t1=None, t0=None):
    """Joint monotone-treatment-response / monotone-treatment-selection
    bounds on E[y(t)], Manski and Pepper (2000).

    Under MTR (each unit response weakly increasing in the ordered
    treatment) and MTS (units selecting higher treatment have weakly
    higher mean response functions), Proposition 3 Corollary 2 of
    Manski-Pepper (2000), eq. (36), gives sharp bounds that need NO
    outcome-support assumption:

        sum_{u<t} E[y|z=u] P(z=u) + E[y|z=t] P(z>=t)
            <= E[y(t)] <=
        sum_{u>t} E[y|z=u] P(z=u) + E[y|z=t] P(z<=t).

    For a treatment-effect contrast ``t1 > t0`` the upper bound is
    ``U(t1) - L(t0)``; the lower bound is 0, because MTR alone already
    implies y(t1) >= y(t0) unit by unit (Manski 1997b, Proposition M2,
    as printed in Molinari 2021 p. 18).  This is the construction of the
    returns-to-schooling application in Section 6 of the paper.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    z : array-like
        Realized treatment on an ordered scale; every distinct value is
        a level.
    t1, t0 : float, optional
        Contrast levels; default the largest and smallest realized
        level.

    Returns
    -------
    RichResult
        ``levels``, ``lower``, ``upper`` (per-level, parallel lists),
        ``ate_lower`` (0), ``ate_upper``, ``t1``, ``t0``, ``n``.

    References
    ----------
    Manski, C. F. and Pepper, J. V. (2000), "Monotone Instrumental
    Variables: With an Application to the Returns to Schooling",
    Econometrica 68(4):997-1010; eq. (36) (Proposition 3, Corollary 2)
    and eq. (34); scanned source
    /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    manski-pepper-2000-monotone-instrumental-variables-returns-to-schooling.pdf
    (NBER TWP 0224, pp. 27-28 of the working paper).
    Manski, C. F. (1997), "Monotone Treatment Response", Econometrica
    65(6):1311-1334, Proposition M2 via Molinari (2021) eq. (2.13),
    local source ~/work/scratch/x000/molinari.pdf.
    """
    yv = C.vec(y)
    zv = C.vec(z)
    n = len(yv)
    if n == 0:
        raise ValueError("bndapp: y is empty")
    if len(zv) != n:
        raise ValueError("bndapp: y and z must have the same length")
    lev = sorted(set(zv))
    pm = []
    for g in lev:
        idx = [i for i in range(n) if zv[i] == g]
        pm.append((len(idx) / float(n),
                   sum(yv[i] for i in idx) / float(len(idx))))
    lower = []
    upper = []
    for k, g in enumerate(lev):
        lo = sum(pm[j][0] * pm[j][1] for j in range(k))
        lo += pm[k][1] * sum(pm[j][0] for j in range(k, len(lev)))
        hi = sum(pm[j][0] * pm[j][1] for j in range(k + 1, len(lev)))
        hi += pm[k][1] * sum(pm[j][0] for j in range(k + 1))
        lower.append(lo)
        upper.append(hi)
    tt1 = lev[-1] if t1 is None else float(t1)
    tt0 = lev[0] if t0 is None else float(t0)
    if tt1 not in lev or tt0 not in lev:
        raise ValueError("bndapp: t1 and t0 must be realized levels of z")
    if not tt1 > tt0:
        raise ValueError("bndapp: need t1 > t0")
    i1 = lev.index(tt1)
    i0 = lev.index(tt0)
    return RichResult(payload={
        "levels": lev, "lower": lower, "upper": upper,
        "ate_lower": 0.0, "ate_upper": upper[i1] - lower[i0],
        "t1": tt1, "t0": tt0, "n": n,
        "method": "Manski-Pepper (2000) MTR-MTS bounds "
                  "(Manski 2007 eqs. 9.18-9.19)"})


# stub-era long name, kept as an alias
bound_application = bndapp


def cheatsheet():
    return ("bndapp: Manski-Pepper MTR-MTS bounds on E[y(t)] "
            "(Manski 2007 eq. 9.18; ATE bound eq. 9.19)")
