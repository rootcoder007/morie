# morie.fn -- function file (rootcoder007/morie)
"""Sample from a GEV distribution.

Implements eq. (3.4) inverted (inverse-CDF sampling) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_sample"]


def evt_gev_sample(n, mu, sigma, xi, seed=42):
    """Inverse-CDF GEV sampling: x = G^{-1}(U), U ~ Uniform(0,1)
    (Coles 2001 eq. 3.4 applied to uniform draws)."""
    rng = np.random.default_rng(seed)
    x = _ev.gev_sample(int(n), float(mu), float(sigma), float(xi), rng)
    res = RichResult(payload={"x": x, "n": int(n),
                              "method": "GEV inverse-CDF sampler (Coles 2001 eq. 3.4)"})
    return with_describe_pointer(res, "evgevs")


def cheatsheet():
    return "evgevs: Sample from a GEV distribution"
