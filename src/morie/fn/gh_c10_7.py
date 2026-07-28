# morie.fn -- function file (rootcoder007/morie)
"""Density estimation via finite random series prior: log f = sum beta_k phi_k."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ghosal_frs_density"]


def ghosal_frs_density(x, grid=None, K=None, s=None, seed=0, n_draws=150):
    r"""Finite random series prior for a density (Ghosal Sec. 10.4.1):

    .. math:: \log f = \sum_{k \le K}\beta_k\varphi_k - \log Z,
              \qquad K \sim \pi_n,

    with the NUMBER of terms itself random.

    That is the entire mechanism for adaptation. A fixed ``K`` gives
    a rate tied to the smoothness the basis can express; putting a
    prior on ``K`` -- with mass decaying fast enough, e.g.
    :math:`\pi(K) \propto e^{-cK\log K}` -- lets the posterior
    concentrate on the ``K`` that matches the truth, attaining
    :math:`n^{-s/(2s+1)}` for every ``s`` WITHOUT knowing ``s``. The
    prior on K is doing the model selection that a fixed basis
    cannot.

    The exponential link again buys positivity and normalisation for
    free, so no constraint has to be imposed on the coefficients.

    Parameters
    ----------
    x : array-like
        Observations.
    grid : array-like, optional
        Evaluation points.
    K : int, optional
        Fixed number of terms; drawn from the prior when omitted,
        which is the adaptive case.
    s : float, optional
        Smoothness to report the rate at.
    seed, n_draws
        RNG seed and Monte Carlo draws over K and the coefficients.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``K_fixed``, ``K_drawn_mean``,
        ``rate``, ``adaptive``, ``prior_on_K``, ``mass``, ``n``,
        ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 10.4 (finite random series) and
    Sec. 10.4.1 (density estimation).
    """
    from ._ghosal import minimax_rate

    xv = np.asarray(x, dtype=float).ravel()
    nn = xv.size
    if nn < 5:
        raise ValueError(f"need at least 5 observations, got {nn}.")
    lo, hi = float(xv.min()), float(xv.max())
    if hi <= lo:
        raise ValueError("the sample has zero spread.")
    g = np.linspace(lo, hi, 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    z = (g - lo) / (hi - lo)
    zx = (xv - lo) / (hi - lo)
    rng = np.random.default_rng(int(seed))
    kmax = max(2, int(np.ceil(nn ** (1.0 / 3.0))))
    # pi(K) proportional to exp(-c K log K): fast enough decay for
    # the sieve remainder condition of the contraction theorem
    ks = np.arange(1, kmax + 1)
    pk = np.exp(-0.5 * ks * np.log(ks + 1))
    pk /= pk.sum()

    dens = np.zeros(g.size)
    kdraw = []
    for _ in range(int(n_draws)):
        kk = int(K) if K is not None else int(rng.choice(ks, p=pk))
        kdraw.append(kk)
        phi_g = np.column_stack([np.cos(np.pi * j * z) for j in range(1, kk + 1)])
        phi_x = np.column_stack([np.cos(np.pi * j * zx) for j in range(1, kk + 1)])
        # coefficients centred on the empirical Fourier coefficients,
        # which is the posterior's location to first order
        beta = phi_x.mean(axis=0) * kk + rng.normal(0, 0.3, kk)
        psi = phi_g @ beta
        f = np.exp(psi - psi.max())
        f /= np.trapezoid(f, g)
        dens += f
    dens /= float(n_draws)
    sv = 1.0 if s is None else float(s)
    return RichResult(payload={
        "grid": g, "density": dens,
        "K_fixed": None if K is None else int(K),
        "K_drawn_mean": float(np.mean(kdraw)),
        "rate": minimax_rate(nn, sv), "adaptive": K is None,
        "prior_on_K": "pi(K) proportional to exp(-c K log K)",
        "mass": float(np.trapezoid(dens, g)), "n": int(nn),
        "method": "Finite random series (Sec. 10.4.1); the prior on K is what makes it adaptive"})


def cheatsheet():
    return "gh_c10_7: the prior on the NUMBER of terms is what buys adaptation to unknown smoothness"
