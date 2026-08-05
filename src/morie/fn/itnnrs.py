# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Item nonresponse adjustment by weighting classes.

Kalton and Flores-Cervantes (2003), "Weighting methods", Journal of
Official Statistics 19(2):81-97.  Their section 2 describes the
weighting-class (response homogeneity group) adjustment: units are
partitioned into classes on observed variables, and within class c the
responding units carry the adjustment factor

    f_c = (sum of base weights in class c)
          / (sum of base weights of respondents in class c),

which is the inverse of the estimated response probability under the
assumption that response is independent of the item value within a
class.  The adjusted estimate is the base-weighted respondent mean with
those factors applied, which is exactly the class-size-weighted average
of the class respondent means.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["item_nonresponse"]


def _classkey(row):
    return "|".join("%.12g" % v for v in row)


def item_nonresponse(y, R, X, weights=None):
    """Weighting-class adjustment for item nonresponse.

    Parameters
    ----------
    y : array-like
        Item values; entries with R = 0 are not used.
    R : array-like
        Response indicator for the item, 0 or 1.
    X : array-like or None
        Class-defining variables, one row per unit.  Rows with equal
        values form one weighting class.  None puts everyone in one class.
    weights : array-like or None
        Base (design) weights.  None means all ones.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("item_nonresponse: y is empty")
    r = core.vec(R)
    if len(r) != n:
        raise ValueError("item_nonresponse: y and R have different lengths")
    for v in r:
        if v not in (0.0, 1.0):
            raise ValueError("item_nonresponse: R must be 0 or 1")
    if X is None:
        keys = ["all"] * n
    else:
        rows = core.mat(X)
        if len(rows) != n:
            raise ValueError("item_nonresponse: X and y have different lengths")
        keys = [_classkey(row) for row in rows]
    d = core.vec(weights) if weights is not None else [1.0] * n
    if len(d) != n:
        raise ValueError("item_nonresponse: weights and y have different lengths")
    tot = {}
    resp = {}
    order = []
    for i in range(n):
        k = keys[i]
        if k not in tot:
            tot[k] = 0.0
            resp[k] = 0.0
            order.append(k)
        tot[k] += d[i]
        if r[i] == 1.0:
            resp[k] += d[i]
    for k in order:
        if resp[k] <= 0:
            raise ValueError("item_nonresponse: weighting class with no respondents")
    fac = {k: tot[k] / resp[k] for k in order}
    num = 0.0
    den = 0.0
    for i in range(n):
        if r[i] == 1.0:
            aw = d[i] * fac[keys[i]]
            num += aw * yv[i]
            den += aw
    est = num / den
    ss = 0.0
    for i in range(n):
        if r[i] == 1.0:
            aw = d[i] * fac[keys[i]]
            ss += (aw * (yv[i] - est)) ** 2
    se = math.sqrt(ss) / den
    rates = [resp[k] / tot[k] for k in order]
    return RichResult(
        title="Item nonresponse adjustment",
        summary_lines=[("classes", len(order)), ("response rate", sum(resp.values()) / sum(tot.values())), ("estimate", est)],
        payload={
            "estimate": est,
            "se": se,
            "n_classes": len(order),
            "response_rate": sum(resp.values()) / sum(tot.values()),
            "min_class_rate": min(rates),
            "max_class_rate": max(rates),
            "adjusted_total": den,
            "n_respondents": int(sum(r)),
            "n": n,
            "method": "weighting-class adjustment f_c = W_c / W_c(resp), Kalton & Flores-Cervantes (2003)",
        },
    )


def cheatsheet():
    return "itnnrs: Item nonresponse adjustment"


# compact alias per ledger/NAMING.md
itemnonresponse = item_nonresponse
