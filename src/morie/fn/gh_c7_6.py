# morie.fn -- function file (rootcoder007/morie)
"""Polya tree density estimation: posterior consistent at Lipschitz densities."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_pt_dens_con"]


def ghosal_pt_dens_con(x, grid=None, levels=6, a_scale=1.0, lo=None, hi=None):
    r"""Polya tree density estimation and its consistency (Ghosal
    Sec. 7.2.3).

    A Polya tree splits the sample space by successive dyadic
    partitions and puts an independent ``Beta(a_m, a_m)`` prior on
    the split at every node of level ``m``. Conjugacy makes the
    posterior split ``Beta(a_m + n_left, a_m + n_right)``, so the
    posterior mean density is a product of expected splits along the
    path to each cell -- closed form, no MCMC.

    **The choice of a_m decides whether the prior is usable at all.**
    With ``a_m`` constant the prior is supported on distributions
    SINGULAR with respect to Lebesgue measure, so no density exists
    to be consistent for. Taking ``a_m`` to grow -- ``m^2`` is the
    standard choice -- puts mass on absolutely continuous laws, and
    the posterior is then Hellinger-consistent at any Lipschitz
    density. ``absolutely_continuous_prior`` reports which regime the
    supplied ``a_m`` is in rather than leaving it to the caller.

    Parameters
    ----------
    x : array-like
        Observations.
    grid : array-like, optional
        Evaluation points.
    levels : int
        Depth of the dyadic partition.
    a_scale : float > 0
        Multiplier on ``a_m = a_scale * m^2``.
    lo, hi : float, optional
        Bounds of the partitioned interval.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``levels``, ``a_rule``,
        ``absolutely_continuous_prior`` (True), ``mass``,
        ``consistent_at`` , ``n``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 3.7 (Polya trees) and Sec. 7.2.3
    (consistency for density estimation).
    """
    from ._ghosal import polya_tree_density

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 4:
        raise ValueError(f"need at least 4 observations, got {xv.size}.")
    sc = float(a_scale)
    if sc <= 0:
        raise ValueError(f"a_scale must be positive, got {sc}.")
    a0 = float(xv.min()) if lo is None else float(lo)
    a1 = float(xv.max()) if hi is None else float(hi)
    g = np.linspace(a0, a1, 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    dens = polya_tree_density(xv, g, levels=levels,
                              a_fn=lambda m: sc * m ** 2, lo=a0, hi=a1)
    return RichResult(payload={
        "grid": g, "density": dens, "levels": int(levels),
        "a_rule": "a_m = a_scale * m^2 (growing, so the prior is on densities)",
        "absolutely_continuous_prior": True,
        "mass": float(np.trapezoid(dens, g)),
        "consistent_at": "any Lipschitz density, in Hellinger distance",
        "n": int(xv.size),
        "method": "Polya tree posterior mean (Sec. 7.2.3); closed form by Beta conjugacy"})


def cheatsheet():
    return "gh_c7_6: constant a_m gives a prior on SINGULAR laws -- a_m must grow, m^2 is standard"
