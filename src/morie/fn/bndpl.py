# morie.fn -- function file (rootcoder007/morie)
"""Polya-tree posterior density."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bnp_density_pl"]


def bnp_density_pl(y, grid=None, tree_depth=6, alpha=1.0, lo=None, hi=None):
    r"""Posterior mean density under a Polya tree prior (Lavine 1992;
    Ferguson 1974).

    On the level-:math:`m` dyadic partition the prior puts
    independent :math:`\text{Beta}(\alpha_m, \alpha_m)` splitting
    probabilities at every node; observing the data updates each by
    conjugacy to :math:`\text{Beta}(\alpha_m + n_L, \alpha_m + n_R)`
    with :math:`n_L, n_R` the counts falling left and right. The
    posterior mean density at a point multiplies the expected splits
    along its path.

    The canonical choice :math:`\alpha_m = \alpha\,m^2` is what makes
    the prior put probability one on ABSOLUTELY CONTINUOUS
    distributions (Kraft 1964); with :math:`\alpha_m` constant the
    tree behaves like a Dirichlet process and its realisations are
    discrete. That distinction is the reason the parameters have the
    shape they do, and it is recorded in the output. Larger
    :math:`\alpha` means more prior smoothing toward the uniform base
    measure; the posterior mean interpolates between the base and the
    empirical histogram as :math:`\alpha` runs from large to small,
    and both limits are tested.

    Computation is shared with ``morie.fn._ghosal.polya_tree_density``
    -- one implementation, two shelves, no drift.

    Parameters
    ----------
    y : array-like, shape (n,)
        Sample.
    grid : array-like, optional
        Evaluation points; a data-driven grid otherwise.
    tree_depth : int, default 6
        Partition levels; resolution is ``2**tree_depth`` cells.
    alpha : float, default 1.0
        The scale in :math:`\alpha_m = \alpha m^2`.
    lo, hi : float, optional
        The support of the base measure; padded data range otherwise.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``mass``, ``tree_depth``,
        ``alpha``, ``alpha_rule``, ``continuity_note``, ``n``,
        ``method``.

    References
    ----------
    Lavine, M. (1992), "Some aspects of Polya tree distributions for
    statistical modelling", *Annals of Statistics* 20:1222-1235.
    Ferguson, T. S. (1974), *Annals of Statistics* 2:615-629. Kraft,
    C. H. (1964), *Journal of Applied Probability* 1:385-388.
    Hanson, T. E. (2006), *JASA* 101:1548-1565.
    """
    from ._ghosal import polya_tree_density

    yv = np.asarray(y, dtype=float).ravel()
    n = yv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    depth = int(tree_depth)
    if not 1 <= depth <= 16:
        raise ValueError(f"tree_depth must lie in 1..16, got {depth}.")
    av = float(alpha)
    if av <= 0:
        raise ValueError(f"alpha must be positive, got {av}.")
    if grid is None:
        pad = 0.05 * (yv.max() - yv.min() + 1e-12)
        grid = np.linspace(yv.min() - pad, yv.max() + pad, 256)
    g = np.atleast_1d(np.asarray(grid, dtype=float))
    dens = polya_tree_density(yv, g, levels=depth,
                              a_fn=lambda m: av * m ** 2, lo=lo, hi=hi)
    dens = np.asarray(dens, dtype=float)
    mass = float(np.trapezoid(dens, g)) if g.size > 2 and \
        np.all(np.diff(g) > 0) else None
    return RichResult(payload={
        "grid": g, "density": dens, "mass": mass,
        "tree_depth": depth, "alpha": av,
        "alpha_rule": "alpha_m = alpha m^2, the Kraft rule that makes the "
                      "prior sit on absolutely continuous distributions; a "
                      "CONSTANT alpha_m gives a Dirichlet-process-like tree "
                      "with discrete realisations",
        "continuity_note": "larger alpha smooths toward the uniform base "
                           "measure; smaller alpha follows the empirical "
                           "histogram",
        "n": int(n),
        "method": "Polya tree posterior mean density (Lavine 1992), "
                  "computed by morie.fn._ghosal.polya_tree_density"})


def cheatsheet():
    return "bndpl: alpha_m = alpha m^2 is what buys absolute continuity -- constant alpha_m is a discrete tree"
