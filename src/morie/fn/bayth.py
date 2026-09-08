# morie.fn -- slice s04 (rootcoder007/morie)
"""Bayes theorem posterior for genomic parameters.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 171-208], Chapter 6, Section 6.1
"Bayes Theorem and Bayesian Linear Regression", p. 171-172.  The section
gives, unnumbered, immediately before equation (6.1):

    f(theta|y) = f(y, theta)/f(y) = f(theta) f(y|theta) / f(y)
               proportional to f(theta) L(theta; y),

"where f(y) = integral f(y|theta) f(theta) d theta = E_theta[f(y|theta)]
is the marginal distribution".  It adds that "once a sample of the
posterior distribution is obtained, estimation of a parameter is often
found by averaging the sample values", which is the posterior mean
reported here.

DETERMINISM.  Nothing is sampled.  The marginal f(y) and the posterior
moments are quadratures on a fixed equally spaced grid by the composite
Simpson rule, which is exact for the polynomial integrands and converges
to machine precision on the conjugate case the anchor uses.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["bayes_theorem_genomic"]


def bayes_theorem_genomic(y, prior_f, likelihood_f, grid=(0.0, 1.0), n_grid=2001):
    """Posterior of theta on a grid, by Bayes theorem.

    Parameters
    ----------
    y : array-like
        The data, passed through to likelihood_f.
    prior_f : callable
        theta -> f(theta), the prior density.
    likelihood_f : callable
        (theta, y) -> L(theta; y), the likelihood.
    grid : pair
        (lo, hi), the support over which theta is integrated.
    n_grid : int
        Number of grid points; forced odd for Simpson's rule.

    Returns
    -------
    estimate   : the posterior mean of theta
    posterior  : the normalised posterior density on the grid
    theta      : the grid
    marginal   : f(y), the normalising constant
    post_var   : the posterior variance
    """
    yy = k.vec(y)
    if not callable(prior_f) or not callable(likelihood_f):
        raise ValueError("bayes_theorem_genomic: prior_f and likelihood_f must be callables")
    lo = float(grid[0])
    hi = float(grid[1])
    if not hi > lo:
        raise ValueError("bayes_theorem_genomic: the grid must have positive width")
    m = int(n_grid)
    if m < 3:
        raise ValueError("bayes_theorem_genomic: n_grid must be at least 3")
    if m % 2 == 0:
        m += 1
    h = (hi - lo) / (m - 1)
    th = [lo + i * h for i in range(m)]
    un = []
    for t in th:
        p = float(prior_f(t))
        if p < 0.0:
            raise ValueError("bayes_theorem_genomic: the prior returned a negative density")
        un.append(p * float(likelihood_f(t, yy)))

    def simpson(v):
        s = v[0] + v[m - 1]
        for i in range(1, m - 1):
            s += (4.0 if i % 2 == 1 else 2.0) * v[i]
        return s * h / 3.0

    marg = simpson(un)
    if not marg > 0.0:
        raise ValueError("bayes_theorem_genomic: the marginal f(y) is not positive")
    post = [v / marg for v in un]
    m1 = simpson([th[i] * post[i] for i in range(m)])
    m2 = simpson([th[i] * th[i] * post[i] for i in range(m)])
    return RichResult(
        title="Bayes theorem posterior",
        summary_lines=[("grid", m), ("n", len(yy))],
        payload={
            "estimate": m1,
            "posterior": post,
            "theta": th,
            "marginal": marg,
            "post_mean": m1,
            "post_var": m2 - m1 * m1,
            "n": len(yy),
            "method": "f(theta|y) = f(theta)L(theta;y)/f(y), f(y) = int f(y|theta)f(theta)dtheta, Chapter 6 Sect. 6.1",
        },
    )


def cheatsheet():
    return "bayth: Bayes theorem posterior for genomic parameters"
