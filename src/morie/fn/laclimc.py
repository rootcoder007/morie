# morie.fn -- function file (rootcoder007/morie)
"""LISA Monte-Carlo significance."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def laclimc(y, W, nsim=99, seed=0):
    r"""LISA Monte-Carlo significance.

    Conditional permutation inference for local Moran's ``I_i`` (Anselin
    1995): holding ``y_i`` fixed, the other ``n - 1`` centred values are
    permuted over the remaining units (the order of Philox uniforms, block
    ``s n + i``), ``I_i*`` recomputed, and the folded pseudo p-value ``(1 +
    min(#{I* >= I_i}, #{I* <= I_i})) / (nsim + 1)`` reported (the GeoDa
    convention; ``spdep::localmoran_perm`` uses R's generator).

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).
    nsim : int
        Permutations per unit.
    seed : int
        Philox seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is the number of units with pseudo p < 0.05;
        ``local_values`` the ``I_i``; ``extra`` has ``p_value``,
        ``mean_sim`` and ``sd_sim`` lists.

    References
    ----------
    Anselin, L. (1995). Local indicators of spatial association -- LISA. *Geographical Analysis*,
    27(2), 93-115.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = laclimc([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W, nsim=19, seed=2)
    >>> r.extra["p_value"]
    [0.5, 0.1, 0.15, 0.4, 0.2, 0.45]
    """
    Ii, p, m, s = lat.local_moran_perm(lat.vec(y), lat.mat(W), int(nsim), seed)
    return SpatialResult(
        name="laclimc",
        statistic=float(sum(1 for v in p if v < 0.05)),
        local_values=Ii,
        extra={"p_value": p, "mean_sim": m, "sd_sim": s},
    )


laclimc_fn = laclimc


def cheatsheet() -> str:
    return "laclimc(y, W, nsim, seed) -> conditional-permutation LISA pseudo p-values"
