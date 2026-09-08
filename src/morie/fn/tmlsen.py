# morie.fn -- function file (rootcoder007/morie)
"""TMLE bias bound under unmeasured confounding."""

from . import _array_core as np

from ._richresult import RichResult
from ._tmle import tmle_ate

__all__ = ["tmle_sensitivity_unmeasured"]


def tmle_sensitivity_unmeasured(y, D, X, gamma_grid=None, trunc=0.01):
    r"""Marginal-sensitivity-model bounds around a TMLE estimate.

    Under Tan's marginal sensitivity model, an unmeasured confounder
    can shift the true propensity odds by at most a factor
    :math:`\Gamma`:

    .. math:: \Gamma^{-1} \le
              \frac{g(W)/(1-g(W))}{g^*(W)/(1-g^*(W))} \le \Gamma.

    The extreme reweightings replace :math:`g` with the two
    :math:`\Gamma`-tilted propensities, and re-running TMLE at each
    gives an interval that contains the estimate for every admissible
    confounder of that strength. The reported ``gamma_critical`` is
    the smallest :math:`\Gamma` at which the interval first covers
    zero -- the direct answer to "how strong would the hidden
    confounder have to be?"

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    gamma_grid : sequence >= 1, optional
        Odds-ratio bounds to sweep. Default 1.0 to 3.0 in 9 steps.
    trunc : float, default 0.01
        Propensity truncation.

    Returns
    -------
    RichResult
        keys: ``gamma`` (grid), ``lower``, ``upper`` (arrays),
        ``ate`` (the Gamma = 1 estimate), ``gamma_critical`` (None if
        the interval never covers zero on the grid), ``n``, ``method``.

    References
    ----------
    Tan, Z. (2006). A distributional approach for causal inference
    using propensity scores. *Journal of the American Statistical
    Association*, 101(476), 1619-1637. (the marginal sensitivity
    model)

    Zhao, Q., Small, D. S. & Bhattacharya, B. B. (2019). Sensitivity
    analysis for inverse probability weighting estimators via the
    percentile bootstrap. *JRSS-B*, 81(4), 735-761.
    """
    grid = np.linspace(1.0, 3.0, 9) if gamma_grid is None else np.asarray(gamma_grid, dtype=float).ravel()
    if np.any(grid < 1):
        raise ValueError("gamma values must be at least 1.")

    base = tmle_ate(y, D, X, trunc=trunc)
    g = base["g"]
    odds = g / (1 - g)

    lows, highs = [], []
    for gam in grid:
        g_lo = np.clip((odds / gam) / (1 + odds / gam), trunc, 1 - trunc)
        g_hi = np.clip((odds * gam) / (1 + odds * gam), trunc, 1 - trunc)
        a = tmle_ate(y, D, X, trunc=trunc, g=g_lo)["ate"]
        b = tmle_ate(y, D, X, trunc=trunc, g=g_hi)["ate"]
        lows.append(min(a, b))
        highs.append(max(a, b))
    lows, highs = np.array(lows), np.array(highs)

    crosses = (lows <= 0) & (highs >= 0)
    gcrit = float(grid[np.argmax(crosses)]) if crosses.any() else None

    return RichResult(
        payload={
            "gamma": grid,
            "lower": lows,
            "upper": highs,
            "ate": base["ate"],
            "gamma_critical": gcrit,
            "n": base["n"],
            "method": "TMLE under the marginal sensitivity model (Gamma-tilted propensities)",
        }
    )


def cheatsheet():
    return "tmlsen: tilt the propensity odds by Gamma both ways; report where zero enters"
