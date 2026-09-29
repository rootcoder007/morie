# morie.fn -- function file (rootcoder007/morie)
"""Geary's C Monte-Carlo test."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lacgmc(y, W, nsim=99, seed=0):
    r"""Geary's C Monte-Carlo test.

    Geary's C of the data against ``nsim`` random permutations of ``y``
    over the units (each permutation the order of ``n`` Philox uniforms,
    block ``s``), with the pseudo p-value ``(1 + #{C* <= C}) / (nsim +
    1)`` for positive autocorrelation (Cliff and Ord 1981, sec. 2.6; as
    ``spdep::geary.mc``, which uses R's generator).

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).
    nsim : int
        Number of permutations.
    seed : int
        Philox seed.

    Returns
    -------
    SpatialResult
        ``statistic`` C, ``p_value``; ``extra`` has ``simulated`` and
        their ``mean``.

    References
    ----------
    Geary, R. C. (1954). The contiguity ratio and statistical mapping. *The Incorporated
    Statistician*, 5(3), 115-145.

    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion, London.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = lacgmc([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W, nsim=19, seed=1)
    >>> round(r.statistic, 12), r.p_value
    (1.284383954155, 0.7)
    """
    C, p, sims = lat.geary_perm(lat.vec(y), lat.mat(W), int(nsim), seed)
    return SpatialResult(
        name="lacgmc", statistic=C, p_value=p, extra={"simulated": sims, "mean": lat.ssum(sims) / len(sims)}
    )


lacgmc_fn = lacgmc


def cheatsheet() -> str:
    return "lacgmc(y, W, nsim, seed) -> Geary's C permutation test"
