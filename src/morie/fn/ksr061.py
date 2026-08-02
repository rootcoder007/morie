# morie.fn -- function file (rootcoder007/morie)
"""Differentiability in quadratic mean."""

from . import _array_core as np

from scipy import integrate

from ._richresult import RichResult

__all__ = ["kosorok_ch3_differentiable_quadratic_mean"]


def kosorok_ch3_differentiable_quadratic_mean(density, score, t_grid=None,
                                              support=(-np.inf, np.inf), theta=0.0):
    r"""Differentiability in quadratic mean (DQM), Kosorok Ch. 3:

    .. math:: \int \Big[\frac{\sqrt{dP_t} - \sqrt{dP}}{t}
              - \tfrac12 g \sqrt{dP}\Big]^2 \to 0
              \quad\text{as } t \to 0.

    DQM is the regularity condition for LAN, and it is deliberately
    stated on the SQUARE ROOT of the density: many families are DQM
    without their densities being pointwise differentiable (the double
    exponential at its kink is the standard example). Requiring
    pointwise differentiability instead would exclude models the theory
    covers.

    Returns the DQM integral along a shrinking t sequence -- it must
    fall toward 0.

    Parameters
    ----------
    density : callable
        ``density(x, theta)`` -> density value.
    score : callable
        ``score(x)`` -> the candidate score function g at theta.
    t_grid : sequence of float, optional
        Shrinking perturbations.
    support : tuple, default (-inf, inf)
        Integration limits.
    theta : float, default 0.0
        Base parameter.

    Returns
    -------
    RichResult
        keys: ``t_grid``, ``dqm_integrals``, ``shrinking``,
        ``score_mean`` (must be ~0), ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3 (differentiability in quadratic mean).
    """
    if t_grid is None:
        t_grid = [0.1, 0.05, 0.02, 0.01]
    t_grid = [float(t) for t in t_grid]
    if any(t <= 0 for t in t_grid):
        raise ValueError("t values must be positive.")
    lo, hi = support

    vals = []
    for t in t_grid:
        def integrand(x, t=t):
            p0 = max(float(density(x, theta)), 0.0)
            pt = max(float(density(x, theta + t)), 0.0)
            term = (np.sqrt(pt) - np.sqrt(p0)) / t - 0.5 * float(score(x)) * np.sqrt(p0)
            return term**2

        v, _ = integrate.quad(integrand, lo, hi, limit=200)
        vals.append(float(v))
    vals = np.array(vals)

    # the score must integrate to zero against the base density
    sm, _ = integrate.quad(
        lambda x: float(score(x)) * max(float(density(x, theta)), 0.0), lo, hi,
        limit=200,
    )
    return RichResult(
        payload={"t_grid": np.array(t_grid), "dqm_integrals": vals,
                 "shrinking": bool(vals[-1] <= vals[0] + 1e-12),
                 "score_mean": float(sm),
                 "method": "DQM on sqrt(density); covers non-differentiable densities"}
    )


def cheatsheet():
    return "ksr061: DQM is on the SQUARE ROOT -- Laplace qualifies despite its kink"
