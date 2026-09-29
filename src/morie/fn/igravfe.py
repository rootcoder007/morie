# morie.fn -- function file (rootcoder007/morie)
"""Gravity model with origin/destination fixed effects."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult


def igravfe(flows, origin_id, dest_id, dist):
    r"""Gravity model with origin/destination fixed effects.

    Structural gravity ``E F_od = exp(a_o + g_d + b log d_od)`` by Poisson
    pseudo-maximum likelihood with origin and destination dummies (the first
    destination level is the reference): the fixed effects absorb the
    multilateral-resistance terms (Anderson and van Wincoop 2003) and PPML
    keeps zero flows (Santos Silva and Tenreyro 2006). Its first-order
    conditions reproduce every origin and destination total, so the fit is
    the doubly constrained model. Equals ``glm(F ~ 0 + factor(o) + factor(d)
    + log(d), family = poisson)``.

    Parameters
    ----------
    flows : array-like, shape (n,)
        Non-negative flows.
    origin_id, dest_id : array-like, shape (n,)
        Origin and destination labels.
    dist : array-like, shape (n,)
        Positive distances.

    Returns
    -------
    SpatialResult
        ``statistic`` is the distance elasticity ``b``; ``extra`` has
        ``se_robust`` (HC0), ``se``, ``origin_effects``,
        ``destination_effects`` (0 for the reference), ``fitted``,
        ``loglik``.

    References
    ----------
    Anderson, J. E. and van Wincoop, E. (2003). Gravity with gravitas. *American Economic Review*,
    93(1), 170-192.

    Santos Silva, J. M. C. and Tenreyro, S. (2006). The log of gravity. *Review of Economics and
    Statistics*, 88(4), 641-658.

    Examples
    --------
    >>> F = [0.0, 4.0, 9.0, 3.0, 0.0, 6.0, 8.0, 5.0, 0.0, 2.0]
    >>> o = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a"]
    >>> dd = ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"]
    >>> dist = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.5]
    >>> round(igravfe([v + 1 for v in F], o, dd, dist).statistic, 10)
    1.9523619098
    """
    F = gv.vec(flows)
    o = list(origin_id.tolist() if hasattr(origin_id, "tolist") else origin_id)
    de = list(dest_id.tolist() if hasattr(dest_id, "tolist") else dest_id)
    d = gv.vec(dist)
    if not (len(F) == len(o) == len(de) == len(d)):
        raise ValueError("all inputs must have the same length")
    ol = sorted(set(o), key=str)
    dl = sorted(set(de), key=str)
    X = [
        [1.0 if o[i] == a else 0.0 for a in ol] + [1.0 if de[i] == b else 0.0 for b in dl[1:]] + [math.log(d[i])]
        for i in range(len(F))
    ]
    g = gv.glm_fit(X, F, "poisson")
    c = g["coefficients"]
    k = len(ol)
    return SpatialResult(
        name="igravfe",
        statistic=c[-1],
        extra={
            "se": g["se"][-1],
            "se_robust": g["se_robust"][-1],
            "origin_effects": dict(zip(ol, c[:k])),
            "destination_effects": dict(zip(dl, [0.0] + c[k:-1])),
            "fitted": g["fitted"],
            "loglik": g["loglik"],
        },
    )


igravfe_fn = igravfe


def cheatsheet() -> str:
    return "igravfe(flows, origin_id, dest_id, dist) -> PPML with origin/destination fixed effects"
