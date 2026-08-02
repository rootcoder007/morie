# morie.fn -- function file (rootcoder007/morie)
"""Smoothed Wilcoxon signed-rank test (Fauzi Ch 5).

Classical Wilcoxon signed-rank: with D_i = X_i - theta_0,
    W = sum_i sign(D_i) * R(|D_i|),  R = rank.
Fauzi's smoothed version replaces the rank by a kernel-smoothed
estimate of n * F_{|D|}(|D_i|):

    W_n = sum_i sign(D_i) * R_smooth(|D_i|),
    R_smooth(t) = sum_j W( (t - |D_j|) / h ).

Under H0:theta=theta0 (symmetric F around theta0),

    E[W_n] = 0,   Var[W_n] = n(n+1)(2n+1)/6,
    z = W_n / sqrt(Var)  ->  N(0, 1).
"""

from . import _array_core as np
from scipy import stats as _sps

from ._richresult import RichResult

__all__ = ["fauzi_smoothed_wilcoxon"]


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


def fauzi_smoothed_wilcoxon(x, theta0=0.0, h=None, alternative="two-sided"):
    """Smoothed Wilcoxon signed-rank test of H0: median(x) = theta0."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "fzwlc -- too few obs"})
    d = x - theta0
    ad = np.abs(d)
    nz = ad > 1e-12
    d = d[nz]
    ad = ad[nz]
    if h is None:
        h = float(_silverman_h(ad))
    n_eff = len(d)
    if n_eff < 5:
        return RichResult(
            payload={"statistic": np.nan, "p_value": np.nan, "n": n_eff, "method": "fzwlc -- too few nonzero"}
        )

    # Smoothed rank: R_smooth(t) = sum_j W((t - |D_j|)/h)
    # We need R(|D_i|) for each i, so vectorise:
    diffs = (ad[:, None] - ad[None, :]) / h  # (n,n)
    R_smooth = np.sum(_sps.norm.cdf(diffs), axis=1)  # length n
    W_n = float(np.sum(np.sign(d) * R_smooth))

    var = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 6.0
    z = W_n / np.sqrt(var)
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
            "statistic": W_n,
            "z": float(z),
            "p_value": float(p),
            "theta0": theta0,
            "h": h,
            "n": n_eff,
            "method": f"Fauzi smoothed Wilcoxon signed-rank ({alternative}) (Ch 5)",
        }
    )


def cheatsheet():
    return "fzwlc: Smoothed Wilcoxon signed-rank test"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(200)
# >>> r = fauzi_smoothed_wilcoxon(x, theta0=0.0)
# >>> r["p_value"] > 0.05
# True
