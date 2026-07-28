# morie.fn -- function file (rootcoder007/morie)
"""Exact posterior predictive for DP: closed-form Polya urn for density estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ghosal_dp_posterior_exact"]


def ghosal_dp_posterior_exact(x, alpha=1.0, grid=None):
    r"""Exact posterior predictive of a Dirichlet process (Ghosal
    Sec. 4.1.4):

    .. math:: p(X_{n+1} \in \cdot \mid X_1,\dots,X_n)
              = \frac{\alpha}{\alpha+n}G_0
              + \frac{1}{\alpha+n}\sum_k n_k \delta_{X_k^{*}}.

    Closed form, no simulation: conjugacy (Sec. 4.1.3) turns the
    posterior of ``F ~ DP(alpha G_0)`` into
    ``DP(alpha G_0 + sum delta_{X_i})``, and the predictive is its
    mean.

    The predictive is NOT a density. It has an atom at every distinct
    observed value carrying total mass ``n/(alpha+n)``, and that is a
    property of the process rather than an approximation: a DP draw
    is almost surely discrete however smooth ``G_0`` is. Anything
    wanting a density must mix the DP, which is what
    :mod:`morie.fn.gh_c5_8` does. ``atom_weight`` is returned so the
    discreteness is visible rather than implied.

    As ``alpha -> 0`` the base measure is ignored and the predictive
    becomes the empirical distribution; as ``alpha -> inf`` it
    becomes ``G_0``. Both limits are returned as ``limit_note``.

    Parameters
    ----------
    x : array-like
        Observations.
    alpha : float > 0
        Concentration of the base measure.
    grid : array-like, optional
        Points at which to report the continuous part.

    Returns
    -------
    RichResult
        keys: ``grid``, ``base_density`` (already weighted),
        ``atoms``, ``atom_probs``, ``base_weight``, ``atom_weight``,
        ``n_distinct``, ``is_density`` (False), ``limit_note``,
        ``n``, ``method``.
    References
    ----------
    Ghosal, S. and van der Vaart, A. *Fundamentals of Nonparametric
    Bayesian Inference*. Cambridge. Sec. 4.1.3-4.1.4.
    """
    from ._ghosal import dp_predictive

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 1:
        raise ValueError("need at least one observation.")
    out = dp_predictive(xv, alpha=alpha, grid=grid)
    return RichResult(payload={
        **out, "is_density": False,
        "limit_note": "alpha -> 0 gives the empirical distribution; "
                      "alpha -> infinity gives G_0",
        "n": int(xv.size),
        "method": "Polya urn predictive (Sec. 4.1.4); atoms at the distinct values, mass n/(alpha+n)"})


def cheatsheet():
    return "gh_dp_post_ex: the predictive is NOT a density -- a DP draw is a.s. discrete"
