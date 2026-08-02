# morie.fn -- function file (rootcoder007/morie)
"""Gaussian location-scale kernel for DPM density estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_gauss_ker"]


def ghosal_gauss_ker(x, grid=None, alpha=1.0, K=50, seed=0, n_draws=200):
    r"""Dirichlet-process mixture of a Gaussian location-scale kernel
    (Ghosal Sec. 5.5):

    .. math:: f(x) = \int \phi_{\mu,\sigma}(x)\,dG(\mu,\sigma),
              \qquad G \sim DP(\alpha G_0).

    Mixing repairs what :mod:`morie.fn.gh_dp_post_ex` exposes: a DP
    draw is discrete, but a discrete mixing distribution over a
    CONTINUOUS kernel gives a continuous density. The location-scale
    version is the workhorse of the chapter because it approximates
    any smooth density well while remaining conjugate enough to fit.

    The estimate here is the prior/posterior mean under a truncated
    stick-breaking representation with ``K`` sticks, averaged over
    ``n_draws`` draws. Truncation is honest rather than incidental --
    the residual mass past K sticks is returned as
    ``truncation_mass`` so it can be checked instead of assumed
    negligible.

    Parameters
    ----------
    x : array-like
        Observations, used to set the base measure and the grid.
    grid : array-like, optional
        Evaluation points.
    alpha : float > 0
        DP concentration.
    K : int
        Stick-breaking truncation level.
    seed : int
        RNG seed.
    n_draws : int
        Monte Carlo draws averaged over.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``alpha``, ``K``,
        ``truncation_mass``, ``is_density`` (True), ``n``,
        ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 5.5 (examples of kernels) and
    Sec. 3.3 (stick-breaking).
    """
    from ._ghosal import stick_breaking

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 2:
        raise ValueError(f"need at least 2 observations, got {xv.size}.")
    a = float(alpha)
    if a <= 0:
        raise ValueError(f"alpha must be positive, got {a}.")
    kk = int(K)
    g = np.linspace(xv.min() - 1, xv.max() + 1, 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    rng = np.random.default_rng(int(seed))
    m0, s0 = float(xv.mean()), float(xv.std(ddof=1))
    if s0 <= 0:
        raise ValueError("the sample has zero spread.")
    dens = np.zeros(g.size)
    lost = 0.0
    for _ in range(int(n_draws)):
        w = stick_breaking(a, kk, rng)
        lost += 1.0 - float(w.sum())
        mu = rng.normal(m0, s0, kk)
        sg = np.abs(rng.normal(s0, 0.3 * s0, kk)) + 1e-6
        comp = np.exp(-0.5 * ((g[:, None] - mu) / sg) ** 2) / \
            (sg * np.sqrt(2 * np.pi))
        dens += comp @ w
    dens /= float(n_draws)
    return RichResult(payload={
        "grid": g, "density": dens, "alpha": a, "K": kk,
        "truncation_mass": float(lost / n_draws),
        "is_density": True, "n": int(xv.size),
        "method": "DP mixture of a Gaussian location-scale kernel (Sec. 5.5), truncated stick-breaking"})


def cheatsheet():
    return "gh_c5_8: a discrete mixing law over a continuous kernel is what restores a density"
