# morie.fn -- function file (rootcoder007/morie)
"""Monte Carlo test of mark dependence by random relabelling (Diggle 2003; Baddeley et al. 2015)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from ._rng import random_uniform
from .mkcorr import mark_correlation

__all__ = ["mark_dependence_test"]


def mark_dependence_test(
    points,
    marks,
    window,
    *,
    nsim: int = 99,
    seed: int = 1,
    statistic: str = "mad",
    rmin: float = 0.0,
    rmax: float | None = None,
    correction: str = "iso",
) -> SpatialResult:
    r"""Test independent marking against the mark correlation function.

    Under random labelling the marks are exchangeable given the locations,
    so the mark correlation ``k_mm(r)`` (:func:`mark_correlation`) of the
    data is compared with those of ``nsim`` random permutations of the
    marks (Diggle 2003, Sec. 7; ``spatstat`` ``mad.test`` / ``dclf.test``
    with ``simulate = rlabel``).  The discrepancy is the largest (``mad``)
    or integrated squared (``dclf``, Diggle 1986; Cressie 1991)
    deviation of ``k_mm`` from 1, its value under independence, over
    ``rmin <= r <= rmax`` (the integral is ``(rmax - rmin)`` times the mean
    squared deviation over the grid, as in spatstat); the p-value is
    ``(1 + #{T_sim >= T_obs}) / (nsim + 1)``.  Permutations are Fisher-Yates
    shuffles driven by Philox stream ``s`` of ``seed``, identical in both
    morie arms.

    :param points: (n, 2) coordinates.
    :param marks: Non-negative numeric marks (n,).
    :param window: ``(xmin, xmax, ymin, ymax)``.
    :param nsim: Number of random relabellings.
    :param seed: Philox seed.
    :param statistic: ``mad`` or ``dclf``.
    :param rmin: Smallest distance in the discrepancy.
    :param rmax: Largest distance (default: the ``markcorr`` range).
    :param correction: ``iso`` or ``trans``.
    :return: :class:`SpatialResult` with ``statistic``, ``p_value`` and
        ``extra`` keys ``simulated``, ``nsim``, ``r``, ``k``.

    References
    ----------
    Diggle, P. J. (1986). Displaced amacrine cells in the retina of a
    rabbit: analysis of a bivariate spatial point pattern. *Journal of
    Neuroscience Methods*, 18, 115-125.
    Diggle, P. J. (2003). *Statistical Analysis of Spatial Point
    Patterns*, 2nd edn. Arnold, London.

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.35, 0.3), (0.8, 0.6), (0.7, 0.15), (0.55, 0.5),
    ...        (0.2, 0.65), (0.9, 0.9), (0.15, 0.95), (0.6, 0.85)]
    >>> mk = [1.0, 2.0, 1.5, 3.0, 0.5, 2.5, 1.0, 2.0, 0.8, 1.2]
    >>> r = mark_dependence_test(pts, mk, (0, 1, 0, 1), nsim=19)
    >>> round(r.statistic, 6), r.p_value
    (0.161439, 1.0)
    """
    if statistic not in ("mad", "dclf"):
        raise ValueError("statistic must be 'mad' or 'dclf'")
    m = [float(v) for v in np.asarray(marks, dtype=float).tolist()]
    n = len(m)
    base = mark_correlation(points, m, window, rmax=rmax, correction=correction)
    r = base.r
    keep = [i for i in range(len(r)) if rmin <= r[i] and base.k[i] == base.k[i]]
    if not keep:
        raise ValueError("no distances in [rmin, rmax] with a defined k_mm")
    span = r[-1] - rmin

    def disc(k):
        dev = [k[i] - 1.0 for i in keep if k[i] == k[i]]
        if statistic == "mad":
            return max(abs(v) for v in dev)
        return span * sum(v * v for v in dev) / len(dev)

    obs = disc(base.k)
    sims = []
    for s in range(int(nsim)):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=s)]
        p = list(m)
        for i in range(n - 1, 0, -1):
            j = min(int(u[i] * (i + 1)), i)
            p[i], p[j] = p[j], p[i]
        sims.append(disc(mark_correlation(points, p, window, r=r, correction=correction).k))
    return SpatialResult(
        name="mark_dependence_test",
        statistic=obs,
        p_value=(1 + sum(1 for t in sims if t >= obs)) / (len(sims) + 1),
        extra={"simulated": sims, "nsim": int(nsim), "r": r, "k": base.k, "method": statistic},
    )


def cheatsheet() -> str:
    return "mark_dependence_test(points, marks, window) -> random-labelling MAD/DCLF test on k_mm."
