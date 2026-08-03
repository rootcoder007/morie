# morie.fn -- function file (rootcoder007/morie)
"""QQ-plot diagnostic for a fitted GEV.

Implements sec. 3.3.4 (quantile plot) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer (equation checked against the
library PDF).
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_qq_plot"]


def evt_gev_qq_plot(x, mu, sigma, xi):
    """Quantile plot pairs (G-hat^{-1}(i/(n+1)), z_(i)) for the fitted
    GEV (Coles 2001 sec. 3.3.4)."""
    xs = sorted(_ev._flat(x))
    n = len(xs)
    q_model = [_ev.gev_quantile((i + 1.0) / (n + 1.0), float(mu),
                                float(sigma), float(xi))
               for i in range(n)]
    res = RichResult(payload={"q_emp": xs, "q_model": q_model, "n": n,
                              "method": "GEV quantile plot (Coles 2001 sec. 3.3.4)"})
    return with_describe_pointer(res, "evqqgev")


def cheatsheet():
    return "evqqgev: QQ-plot diagnostic for a fitted GEV"
