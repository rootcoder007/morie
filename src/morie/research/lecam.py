# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P1 (continued): Le Cam's two-point lower bound (``research/lean/P1LeCam.lean``; Le Cam 1973; Yu 1997).

* ``Research.P1LeCam.sum_min`` / ``tv_nonneg`` / ``tv_le_one``
* ``Research.P1LeCam.two_point`` / ``minimax``

R parity: ``rmorie`` ``R/lecam_bound.R`` (``morie_two_point_bound``).
"""

from __future__ import annotations

import math

__all__ = ["two_point_bound"]


def two_point_bound(p, q, theta_p, theta_q, estimator=None) -> dict:
    """Le Cam's two-point lower bound on a finite sample space.

    Examples
    --------
    >>> p = [0.0625, 0.25, 0.375, 0.25, 0.0625]
    >>> q = [0.0256, 0.1536, 0.3456, 0.3456, 0.1296]
    >>> b = two_point_bound(p, q, theta_p=2, theta_q=3, estimator=[0, 1.25, 2.5, 3.75, 5])
    >>> (round(b["tv"], 12), round(b["bound"], 12), round(b["minimax_risk"], 12), b["satisfied"], round(b["sum_min"], 12))
    (0.1627, 0.41865, 1.125, True, 0.8373)
    """
    try:
        pp = [float(v) for v in p]
        qq = [float(v) for v in q]
    except (TypeError, ValueError) as exc:
        raise ValueError("p and q must be numeric of equal length without NA") from exc
    if len(pp) != len(qq) or any(math.isnan(v) for v in pp + qq):
        raise ValueError("p and q must be numeric of equal length without NA")
    if any(v < 0 for v in pp + qq) or abs(math.fsum(pp) - 1) > 1e-8 or abs(math.fsum(qq) - 1) > 1e-8:
        raise ValueError("p and q must be probability vectors")
    if (
        isinstance(theta_p, list | tuple)
        or isinstance(theta_q, list | tuple)
        or math.isnan(theta_p)
        or math.isnan(theta_q)
    ):
        raise ValueError("theta_p and theta_q must be single numbers")
    tv = math.fsum(abs(a - b) for a, b in zip(pp, qq)) / 2
    delta = abs(theta_p - theta_q)
    out = {
        "tv": tv,
        "sum_min": math.fsum(min(a, b) for a, b in zip(pp, qq)),
        "delta": delta,
        "bound": delta * (1 - tv) / 2,
    }
    if estimator is not None:
        T = [float(v) for v in estimator]
        if len(T) != len(pp) or any(math.isnan(v) for v in T):
            raise ValueError("estimator must give one value per point of the space")
        rp = math.fsum(a * abs(t - theta_p) for a, t in zip(pp, T))
        rq = math.fsum(b * abs(t - theta_q) for b, t in zip(qq, T))
        out["risk_p"] = rp
        out["risk_q"] = rq
        out["minimax_risk"] = max(rp, rq)
        out["satisfied"] = max(rp, rq) >= out["bound"] - 1e-12
    out["theorems"] = [
        "Research.P1LeCam.sum_min",
        "Research.P1LeCam.tv_le_one",
        "Research.P1LeCam.two_point",
        "Research.P1LeCam.minimax",
    ]
    return out
