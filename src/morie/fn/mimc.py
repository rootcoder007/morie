# morie.fn -- function file (rootcoder007/morie)
"""Moran's I Monte-Carlo permutation test."""

from ._containers import SpatialResult
from .spregx import moran_permutation_test


def mimc(y, W, nsim=99, seed=0):
    r"""Moran's I Monte-Carlo permutation test (``spdep::moran.mc``).

    The values are permuted ``nsim`` times with Philox-driven Fisher-Yates
    shuffles (stream ``k`` of ``seed`` for permutation ``k``, identical in the
    R arm) and the one-sided pseudo p-value is ``(1 + #{I_sim >= I}) / (nsim +
    1)``. Thin front-end to :func:`morie.fn.spregx.moran_permutation_test`.

    References
    ----------
    Hope, A. C. A. (1968). A simplified Monte Carlo significance test
    procedure. *J. R. Stat. Soc. B* 30, 582-598.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(mimc([1.0, 2.0, 3.0, 4.0], W, nsim=19).statistic, 12)
    0.333333333333
    """
    r = moran_permutation_test(y, W, nsim=int(nsim), seed=int(seed))
    sims = list(r["simulated"])
    return SpatialResult(
        name="mimc",
        statistic=r["statistic"],
        p_value=r["p_value"],
        extra={"simulated": sims, "nsim": int(nsim), "seed": int(seed)},
    )


mimc_fn = mimc


def cheatsheet() -> str:
    return "mimc(y, W, nsim=99, seed=0) -> Moran's I permutation test, pseudo p = (1 + #{I_sim >= I})/(nsim + 1)."
