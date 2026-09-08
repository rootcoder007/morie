# morie.fn -- slice s03 (rootcoder007/morie)
"""Inverse-distance weighting.

Source consulted: Shepard, D. (1968).  A two-dimensional interpolation
function for irregularly-spaced data.  *ACM National Conference* 23,
517-524, whose interpolant is

    z(s*) = sum_i w_i z_i / sum_i w_i,   w_i = 1 / d(s*, s_i)^p

with the convention that if s* coincides with a datum the interpolated
value *is* that datum -- Shepard's function is an exact interpolator,
and the limit is taken rather than the division attempted.  The 1968
proceedings were not retrievable here; the interpolant and the
exactness convention are quoted in their standard published form.

p = 2 is Shepard's own choice and is the default.  The effective number
of contributing points, (sum w)^2 / sum w^2, is returned because it is
the honest measure of how local the estimate really is.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["inverse_distance_weighting"]


def inverse_distance_weighting(coords, values, s_predict=None, power=2.0):
    """IDW interpolation at one or more prediction locations.

    Returns
    -------
    estimate : the value at the first prediction location
    pred     : values at every prediction location
    ess      : effective number of contributing points, per location
    exact    : whether the location coincided with a datum
    """
    P = k.mat(coords)
    z = k.vec(values)
    S = k.mat(s_predict) if s_predict is not None else P
    p = float(power)
    out = []
    ess = []
    exact = []
    for t in range(len(S)):
        w = []
        hit = -1
        for i in range(len(P)):
            s = 0.0
            for a in range(len(P[i])):
                d = P[i][a] - S[t][a]
                s += d * d
            d = math.sqrt(s)
            if d == 0.0:
                hit = i
            w.append(1.0 / (d ** p) if d > 0.0 else 0.0)
        if hit >= 0:
            out.append(z[hit])
            ess.append(1.0)
            exact.append(True)
            continue
        sw = 0.0
        sw2 = 0.0
        num = 0.0
        for i in range(len(P)):
            sw += w[i]
            sw2 += w[i] * w[i]
            num += w[i] * z[i]
        out.append(num / sw if sw > 0.0 else float("nan"))
        ess.append((sw * sw) / sw2 if sw2 > 0.0 else float("nan"))
        exact.append(False)
    return RichResult(
        title="Inverse-distance weighting",
        summary_lines=[("power", p), ("locations", len(S))],
        payload={
            "estimate": out[0] if out else float("nan"),
            "pred": out,
            "ess": ess,
            "exact": exact,
            "power": p,
            "method": "Shepard (1968) inverse-distance weighting, exact at the data",
        },
    )


def cheatsheet():
    return "idwsmt: Inverse distance weighting interpolation"
