# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic properties of the kernel mean residual life (MRL) (Fauzi Ch 4).

For non-negative X with survival S(t)=1-F(t), the MRL is

    m(t) = E[X - t | X > t] = (1/S(t)) * integral_t^inf S(u) du.

The kernel estimator plugs in the kernel-smoothed survival and
integrates.  Asymptotically:

    sqrt(n) (m_hat - m) -> N(0, sigma_m^2(t)),

with sigma_m^2(t) computed via the Yang (1978) plug-in.
"""

import numpy as np
from scipy import stats as _sps

from ._richresult import RichResult

__all__ = ["fauzi_mrl_asymptotic"]


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


def fauzi_mrl_asymptotic(x, t=None, h=None):
    """Kernel MRL estimate at ``t`` with asymptotic SE."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n, "method": "fzmrl -- too few obs"})
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))

    S_t = float(np.mean(_sps.norm.sf((t - x) / h)))
    if S_t <= 0:
        return RichResult(
            payload={"estimate": np.nan, "se": np.nan, "S_hat": S_t, "n": n, "t": t, "method": "fzmrl -- S(t)=0"}
        )

    diffs = x - t
    above = diffs > 0
    if not above.any():
        return RichResult(
            payload={"estimate": 0.0, "se": np.nan, "S_hat": S_t, "n": n, "t": t, "method": "fzmrl -- no x>t"}
        )
    m_hat = float(np.mean(diffs[above]))
    second = float(np.mean((diffs[above]) ** 2))
    sigma2 = (second - m_hat * m_hat) / S_t
    sigma2 = max(sigma2, 0.0)
    se = float(np.sqrt(sigma2 / n))

    return RichResult(
        payload={
            "estimate": m_hat,
            "se": se,
            "S_hat": S_t,
            "t": t,
            "h": h,
            "n": n,
            "method": "Fauzi kernel MRL asymptotic (Ch 4)",
        }
    )


def cheatsheet():
    return "fzmrl: Kernel mean residual life + Yang asymptotic SE"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.exponential(scale=1.0, size=2000)
# >>> r = fauzi_mrl_asymptotic(x, t=0.0)
# >>> abs(r["estimate"] - 1.0) < 0.15  # MRL(0)=1 for Exp(1)
# True
