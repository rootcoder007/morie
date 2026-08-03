# morie.fn -- function file (rootcoder007/morie)
"""PP-plot diagnostic for a fitted GEV.

Implements sec. 3.3.4 (probability plot) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_pp_plot"]


def evt_gev_pp_plot(x, mu, sigma, xi):
    """Probability plot pairs (i/(n+1), G-hat(z_(i))) for the fitted
    GEV (Coles 2001 sec. 3.3.4): near the diagonal when the model
    fits."""
    xs = sorted(_ev._flat(x))
    n = len(xs)
    p_emp = [(i + 1.0) / (n + 1.0) for i in range(n)]
    p_model = [_ev.gev_cdf(v, float(mu), float(sigma), float(xi))
               for v in xs]
    res = RichResult(payload={"p_emp": p_emp, "p_model": p_model,
                              "n": n,
                              "method": "GEV probability plot (Coles 2001 sec. 3.3.4)"})
    return with_describe_pointer(res, "evppgev")


def cheatsheet():
    return "evppgev: PP-plot diagnostic for a fitted GEV"
