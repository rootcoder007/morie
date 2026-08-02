# morie.fn -- function file (rootcoder007/morie)
"""Kolmogorov-Smirnov test with kernel-smoothed CDF (Fauzi Ch 5).

    D_n = sup_t | F_hat_h(t) - F_0(t) |

Under H0:X~F_0, sqrt(n) D_n has the same Kolmogorov limit as the
classical KS, so we use SciPy's Kolmogorov tail.
"""

from . import _array_core as np
from scipy import stats as _sps

from ._richresult import RichResult

__all__ = ["fauzi_ks_smoothed"]


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


def fauzi_ks_smoothed(x, cdf="norm", args=None, h=None, n_grid=512):
    """Smoothed KS test of H0: X ~ ``cdf(*args)``.

    Parameters
    ----------
    x : array-like
    cdf : str or callable    SciPy dist name (default "norm") or F_0(t).
    args : tuple, optional   distribution params; default = MLE on x.
    h : float, optional      bandwidth.
    n_grid : int             grid resolution for sup-evaluation.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "fzksm -- too few obs"})
    if h is None:
        h = float(_silverman_h(x))

    if callable(cdf):
        F0 = cdf
    else:
        dist = getattr(_sps, cdf)
        if args is None:
            if cdf == "norm":
                args = (float(np.mean(x)), float(np.std(x, ddof=1)))
            else:
                args = ()

        def F0(t, dist=dist, args=args):
            return dist.cdf(t, *args)

    lo = float(np.min(x)) - 6 * h
    hi = float(np.max(x)) + 6 * h
    grid = np.linspace(lo, hi, n_grid)
    F_hat = np.array([np.mean(_sps.norm.cdf((g - x) / h)) for g in grid])
    F_ref = np.array([F0(g) for g in grid])
    D_n = float(np.max(np.abs(F_hat - F_ref)))

    p = float(_sps.kstwobign.sf(np.sqrt(n) * D_n))

    return RichResult(
        payload={
            "statistic": D_n,
            "p_value": p,
            "h": h,
            "n": n,
            "method": "Fauzi kernel-smoothed KS test (Ch 5)",
        }
    )


def cheatsheet():
    return "fzksm: Kernel-smoothed Kolmogorov-Smirnov GoF test"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_ks_smoothed(x, cdf="norm", args=(0.0, 1.0))
# >>> r["p_value"] > 0.05
# True
