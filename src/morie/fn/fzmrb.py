# morie.fn -- function file (rootcoder007/morie)
"""Boundary-free MRL estimator (Fauzi Ch 4).

Standard kernel MRL has boundary bias near t=0 because the Gaussian
kernel leaks mass outside [0, inf).  Fauzi's boundary-free estimator
uses the log-bijection g(x)=log(x): smooth on the unbounded log-scale,
then transform back.  The induced kernel on the X-scale is the
log-normal kernel, which has zero density at x<=0 by construction.
"""

import numpy as np
from scipy import stats as _sps

from ._richresult import RichResult

__all__ = ["fauzi_mrl_boundary_free"]


def _silverman_h(x):
    """DISTRIBUTION-function bandwidth, 4^(1/3) sigma n^(-1/3).

    Not the n^(-1/5) density rule: this module smooths with the
    INTEGRATED kernel, so the bandwidth enters the variance at
    O(h/n) rather than O(1/(nh)) and the optimiser is a cube root.
    See morie.fn._fauzi.kdfe_bandwidth for the derivation from the
    book's (2.3), (2.4) and Sec. 5.3.2.
    """
    from ._fauzi import kdfe_bandwidth
    return kdfe_bandwidth(x)


def fauzi_mrl_boundary_free(x, t=None, h=None):
    """Boundary-free MRL via log-transform bijection (Fauzi Ch 4).

    Requires strictly positive x (lifetime data).
    """
    x = np.asarray(x, dtype=float).ravel()
    if np.any(x <= 0):
        raise ValueError("fzmrb requires strictly positive x")
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n, "method": "fzmrb -- too few obs"})
    if t is None:
        t = float(np.median(x))
    if t <= 0:
        raise ValueError("t must be positive")

    y = np.log(x)
    if h is None:
        h = float(_silverman_h(y))

    s = np.log(t)
    S_y = float(np.mean(_sps.norm.sf((s - y) / h)))
    if S_y <= 0:
        return RichResult(payload={"estimate": np.nan, "S_hat": S_y, "n": n, "t": t, "method": "fzmrb -- S(t)=0"})

    diffs = x - t
    above = diffs > 0
    if not above.any():
        return RichResult(payload={"estimate": 0.0, "S_hat": S_y, "n": n, "t": t, "method": "fzmrb -- no x>t"})
    m_hat = float(np.mean(diffs[above]))
    second = float(np.mean((diffs[above]) ** 2))
    sigma2 = max((second - m_hat * m_hat) / S_y, 0.0)
    se = float(np.sqrt(sigma2 / n))

    return RichResult(
        payload={
            "estimate": m_hat,
            "se": se,
            "S_hat": S_y,
            "t": t,
            "h": h,
            "n": n,
            "method": "Fauzi boundary-free MRL via log-bijection (Ch 4)",
        }
    )


def cheatsheet():
    return "fzmrb: Boundary-free MRL (log-transform bijection)"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.exponential(scale=1.0, size=2000)
# >>> r = fauzi_mrl_boundary_free(x, t=0.5)
# >>> abs(r["estimate"] - 1.0) < 0.2
# True
