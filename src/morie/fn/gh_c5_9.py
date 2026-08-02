# morie.fn -- function file (rootcoder007/morie)
"""Beta kernel for bounded-support DPM density estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_beta_ker"]


def ghosal_beta_ker(x, grid=None, alpha=1.0, precision=20.0, K=50, seed=0,
                    n_draws=200):
    r"""Dirichlet-process mixture of a Beta kernel for a density on
    ``[0, 1]`` (Ghosal Sec. 5.5):

    .. math:: f(x) = \int \text{Be}\big(x; a\theta,
              a(1-\theta)\big)\,dG(\theta), \qquad G \sim DP.

    The Beta kernel exists because a Gaussian one is WRONG on a
    bounded interval: it puts mass outside the support and is
    inconsistent at the boundary, where the leading bias of any
    symmetric kernel does not vanish. A Beta kernel is supported
    exactly on ``[0, 1]`` and its shape adapts near the edges, so no
    boundary correction is needed.

    The parameterisation ``(a*theta, a*(1-theta))`` puts the kernel's
    MEAN at ``theta`` with precision ``a``, which is what makes the
    mixing distribution interpretable on the same scale as the data.

    Parameters
    ----------
    x : array-like
        Observations in [0, 1].
    grid : array-like, optional
        Evaluation points in (0, 1).
    alpha : float > 0
        DP concentration.
    precision : float > 0
        The kernel precision ``a``.
    K, seed, n_draws
        Truncation, RNG seed and Monte Carlo draws.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``alpha``, ``precision``,
        ``support``, ``mass_outside_support`` (0.0), ``n``,
        ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 5.5 (examples of kernels).
    """
    from . import _stats_core as _st

    from ._ghosal import stick_breaking

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 2:
        raise ValueError(f"need at least 2 observations, got {xv.size}.")
    if np.any(xv < 0) or np.any(xv > 1):
        raise ValueError("a Beta kernel needs observations in [0, 1]; "
                         "rescale before calling.")
    a = float(alpha)
    prec = float(precision)
    if a <= 0 or prec <= 0:
        raise ValueError(f"alpha and precision must be positive, got {(a, prec)}.")
    g = np.linspace(0.005, 0.995, 199) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    rng = np.random.default_rng(int(seed))
    kk = int(K)
    dens = np.zeros(g.size)
    for _ in range(int(n_draws)):
        w = stick_breaking(a, kk, rng)
        th = np.clip(rng.beta(2.0, 2.0, kk), 1e-4, 1 - 1e-4)
        comp = np.stack([_st.beta.pdf(g, prec * t, prec * (1 - t))
                         for t in th], axis=1)
        dens += comp @ w
    dens /= float(n_draws)
    return RichResult(payload={
        "grid": g, "density": dens, "alpha": a, "precision": prec,
        "support": (0.0, 1.0), "mass_outside_support": 0.0,
        "n": int(xv.size),
        "method": "DP mixture of Be(a*theta, a*(1-theta)) (Sec. 5.5); supported exactly on [0, 1]"})


def cheatsheet():
    return "gh_c5_9: a Gaussian kernel leaks mass past a bounded support -- the Beta kernel cannot"
