# morie.fn -- function file (rootcoder007/morie)
"""Kernel survival function estimator (Fauzi Ch 4).

For non-negative X:

    S_hat_h(t) = 1 - F_hat_h(t) = (1/n) * sum_i ( 1 - W((t-X_i)/h) ),

with W(u) = Phi(u) for the Gaussian kernel.  Asymptotic variance is
the KDFE variance: S(t)(1-S(t))/n + O(h/n).
"""

import numpy as np
from scipy import stats as _sps

from ._richresult import RichResult

__all__ = ["fauzi_survival_kernel"]


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


def fauzi_survival_kernel(x, t=None, h=None):
    """Kernel survival estimate at ``t`` with asymptotic 95 percent CI."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n, "method": "fzsrv -- too few obs"})
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))

    F_hat = float(np.mean(_sps.norm.cdf((t - x) / h)))
    S_hat = 1.0 - F_hat
    se = float(np.sqrt(S_hat * (1.0 - S_hat) / n))
    z = 1.959963984540054
    lo = max(0.0, S_hat - z * se)
    hi = min(1.0, S_hat + z * se)

    return RichResult(
        payload={
            "estimate": S_hat,
            "se": se,
            "ci_lower": lo,
            "ci_upper": hi,
            "t": t,
            "h": h,
            "n": n,
            "method": "Fauzi kernel survival S_hat(t)=1-F_hat_h(t) (Ch 4)",
        }
    )


def cheatsheet():
    return "fzsrv: Kernel survival S_hat(t) = 1 - F_hat_h(t) + 95% CI"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.exponential(scale=1.0, size=2000)
# >>> r = fauzi_survival_kernel(x, t=1.0)
# >>> abs(r["estimate"] - np.exp(-1)) < 0.05  # S(1)=e^-1
# True
