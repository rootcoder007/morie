# morie.fn -- function file (rootcoder007/morie)
"""Parameter-stability plot for GPD threshold choice.

Implements sec. 4.3.4 (modified scale, eq. 4.9 context) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_param_stability"]


def evt_param_stability(x, u_grid=None):
    """Fit the GPD above each threshold in ``u_grid`` and track the
    modified scale sigma* = sigma_u - xi u and the shape xi: both are
    constant in u where the GPD is valid, because sigma_u = sigma_u0 +
    xi u (Coles 2001 eq. 4.9 and sec. 4.3.4)."""
    xs = sorted(_ev._flat(x))
    n = len(xs)
    if u_grid is None:
        qs = [0.5 + 0.4 * k / 9.0 for k in range(10)]
        u_grid = [xs[min(int(q * n), n - 1)] for q in qs]
    sig_star, xis, used = [], [], []
    for u in u_grid:
        exc = [v - u for v in xs if v > u]
        if len(exc) < 10:
            continue
        f = _ev.gpd_mle(exc)
        sig_star.append(f["sigma"] - f["xi"] * float(u))
        xis.append(f["xi"])
        used.append(float(u))
    res = RichResult(payload={"u_grid": used, "sigma_star": sig_star,
                              "xi": xis,
                              "method": "GPD parameter-stability plot (Coles 2001 sec. 4.3.4)"})
    return with_describe_pointer(res, "evprmstab")


def cheatsheet():
    return "evprmstab: Parameter-stability plot for GPD threshold choice"
