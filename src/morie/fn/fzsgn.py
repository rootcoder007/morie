# morie.fn -- function file (rootcoder007/morie)
"""Smoothed sign test (Fauzi Ch 5).

Classical sign test counts S = #{X_i > theta_0}.  Fauzi's smoothed
version replaces the indicator with the integrated kernel:

    S_n = sum_i W( (X_i - theta_0) / h ),   W = integral K.

Under H0: median(X)=theta_0,  E[S_n]=n/2, Var[S_n]≈n/4, so
z = (S_n - n/2)/sqrt(n/4) ~ N(0,1).
"""

from . import _array_core as np
from . import _stats_core as _sps

from ._richresult import RichResult

__all__ = ["fauzi_smoothed_sign"]


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


def fauzi_smoothed_sign(x, theta0=0.0, h=None, alternative="two-sided"):
    """Smoothed sign test of H0: median = theta0."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "fzsgn -- too few obs"})
    if h is None:
        h = float(_silverman_h(x))

    S_n = float(np.sum(_sps.norm.cdf((x - theta0) / h)))
    z = (S_n - n / 2.0) / np.sqrt(n / 4.0)

    if alternative == "two-sided":
        p = 2.0 * (1.0 - _sps.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1.0 - _sps.norm.cdf(z)
    elif alternative == "less":
        p = float(_sps.norm.cdf(z))
    else:
        raise ValueError("alternative must be two-sided/greater/less")

    return RichResult(
        payload={
            "statistic": S_n,
            "z": float(z),
            "p_value": float(p),
            "theta0": theta0,
            "h": h,
            "n": n,
            "method": f"Fauzi smoothed sign test ({alternative}) (Ch 5)",
        }
    )


def cheatsheet():
    return "fzsgn: Smoothed sign test for the median"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_smoothed_sign(x, theta0=0.0)
# >>> r["p_value"] > 0.05
# True
