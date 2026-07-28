# morie.fn -- function file (rootcoder007/morie)
"""Mixtures of Polya tree processes as prior for density estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ghosal_mpt_prior"]


def ghosal_mpt_prior(x, grid=None, levels=6, a_scale=1.0, shifts=None):
    r"""Mixture of Polya trees (Ghosal Sec. 3.7.2):

    .. math:: f \sim \int PT(\alpha, \pi)\,dH(\alpha, \pi),

    a Polya tree with the PARTITION itself given a prior.

    This exists to remove an artefact, not to add flexibility. A
    Polya tree is tied to a fixed dyadic partition, and its posterior
    mean density has visible DISCONTINUITIES at the partition
    boundaries -- an artefact of the tessellation rather than of the
    data. Mixing over the partition (here, over its location)
    averages those breaks away, and the result is smooth while
    keeping the closed-form conjugacy of each component.

    The returned ``max_jump`` measures the largest neighbouring-cell
    jump in the mixed density, and ``max_jump_single`` the same for
    one component, so the smoothing is a measurement rather than a
    claim.

    Parameters
    ----------
    x : array-like
        Observations.
    grid : array-like, optional
        Evaluation points.
    levels : int
        Depth of each dyadic partition.
    a_scale : float > 0
        Multiplier on ``a_m = a_scale * m^2``.
    shifts : array-like, optional
        Partition offsets mixed over, as fractions of a cell width.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``n_components``, ``max_jump``,
        ``max_jump_single``, ``smoother_than_single`` , ``n``,
        ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 3.7.2 (mixtures of Polya tree
    processes).
    """
    from ._ghosal import polya_tree_density

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 4:
        raise ValueError(f"need at least 4 observations, got {xv.size}.")
    sc = float(a_scale)
    if sc <= 0:
        raise ValueError(f"a_scale must be positive, got {sc}.")
    lo, hi = float(xv.min()), float(xv.max())
    span = hi - lo
    if span <= 0:
        raise ValueError("the sample has zero spread.")
    g = np.linspace(lo, hi, 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    sh = np.linspace(0.0, 0.5, 8) if shifts is None else \
        np.atleast_1d(np.asarray(shifts, dtype=float))
    cell = span / (2 ** int(levels))
    comps = []
    for s in sh:
        off = float(s) * cell
        comps.append(polya_tree_density(xv, g, levels=levels,
                                        a_fn=lambda m: sc * m ** 2,
                                        lo=lo - off, hi=hi + (cell - off)))
    comps = np.array(comps)
    dens = comps.mean(axis=0)
    return RichResult(payload={
        "grid": g, "density": dens, "n_components": int(sh.size),
        "max_jump": float(np.max(np.abs(np.diff(dens)))),
        "max_jump_single": float(np.max(np.abs(np.diff(comps[0])))),
        "smoother_than_single": bool(
            np.max(np.abs(np.diff(dens))) <=
            np.max(np.abs(np.diff(comps[0])))),
        "n": int(xv.size),
        "method": "Mixture of Polya trees (Sec. 3.7.2); averages away the partition artefacts"})


def cheatsheet():
    return "gh_c3_14: mixing over the partition removes the tessellation's jumps, not a lack of flexibility"
