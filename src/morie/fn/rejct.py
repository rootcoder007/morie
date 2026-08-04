# morie.fn -- k02 batch (rootcoder007/morie)
"""Hampel's rejection point, gross-error and local-shift sensitivities.

Source consulted: Hampel, F.R. (1974), The influence curve and its role in
robust estimation, *JASA* 69(346), 383-393, section 2, which defines for an
M-estimator with score psi, at the standard normal:

    rejection point   rho* = inf { r > 0 : psi(x) = 0 for all |x| > r }
    gross-error sens. gamma* = sup |psi| / E[psi']
    local-shift sens. lambda* = sup |psi(x) - psi(y)| / |x - y|

For Huber's psi_k the rejection point is infinite (it never redescends),
gamma* = k / (2 Phi(k) - 1) and lambda* = 1.  For Tukey's biweight
psi_c(x) = x (1 - (x/c)^2)^2 on |x| <= c the rejection point is exactly c,
sup|psi| = 16 c / (25 sqrt 5) at x = c/sqrt 5, and E[psi'] is obtained from
Simpson's rule on [-c, c] with a fixed panel count, so both arms integrate
identically.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as _st

from ._richresult import RichResult

__all__ = ["rejection_point"]

_INF = float("inf")


def _simpson(f, a, b, panels):
    h = (b - a) / panels
    tot = f(a) + f(b)
    for i in range(1, panels):
        tot += (4.0 if i % 2 == 1 else 2.0) * f(a + i * h)
    return tot * h / 3.0


def rejection_point(psi="huber", tuning=None, panels=2000):
    """Hampel's three robustness measures for a score function.

    Parameters
    ----------
    psi : {"huber", "bisquare"}, default "huber"
        Score family.
    tuning : float, optional
        k for Huber (default 1.345), c for the biweight (default 4.685).
    panels : int, default 2000
        Even number of Simpson panels for E[psi'] in the biweight case.

    Returns
    -------
    RichResult
        estimate (rejection point), gross_error_sensitivity,
        local_shift_sensitivity, sup_psi, expected_psi_prime, tuning,
        psi, n, method.
    """
    fam = str(psi).lower()
    if fam == "huber":
        k = 1.345 if tuning is None else float(tuning)
        eprime = 2.0 * float(_st.norm.cdf(k)) - 1.0
        sup = k
        return RichResult(
            payload={
                "estimate": _INF,
                "gross_error_sensitivity": float(sup / eprime),
                "local_shift_sensitivity": 1.0,
                "sup_psi": float(sup),
                "expected_psi_prime": float(eprime),
                "tuning": float(k),
                "psi": "huber",
                "n": 0,
                "method": "Hampel robustness measures for Huber's psi (Hampel 1974, sec. 2)",
            }
        )
    if fam not in ("bisquare", "biweight", "tukey"):
        raise ValueError("psi must be 'huber' or 'bisquare'")
    c = 4.685 if tuning is None else float(tuning)
    phi = lambda t: float(_st.norm.pdf(t))
    dpsi = lambda t: (1.0 - 6.0 * t * t / (c * c) + 5.0 * t**4 / c**4) * phi(t)
    eprime = _simpson(dpsi, -c, c, int(panels))
    sup = 16.0 * c / (25.0 * float(np.sqrt(5.0)))
    return RichResult(
        payload={
            "estimate": float(c),
            "gross_error_sensitivity": float(sup / eprime),
            "local_shift_sensitivity": 1.0,
            "sup_psi": float(sup),
            "expected_psi_prime": float(eprime),
            "tuning": float(c),
            "psi": "bisquare",
            "n": int(panels),
            "method": "Hampel robustness measures for Tukey's biweight (Hampel 1974, sec. 2)",
        }
    )


# CANONICAL TEST
# >>> r = rejection_point("huber", 1.345)
# >>> assert r["estimate"] == float("inf")        # Huber's psi never redescends
# >>> assert abs(r["gross_error_sensitivity"] - 1.345 / 0.8213378) < 1e-5
# >>> b = rejection_point("bisquare", 4.685)
# >>> assert abs(b["estimate"] - 4.685) < 1e-15   # rejection point IS c
# >>> assert abs(b["sup_psi"] - 16 * 4.685 / (25 * 5 ** 0.5)) < 1e-14


def cheatsheet():
    return "rejct(psi, tuning): Hampel rejection point and sensitivities."


rejectionpoint = rejection_point
