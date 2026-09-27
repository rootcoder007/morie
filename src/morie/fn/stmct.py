# morie.fn -- function file (rootcoder007/morie)
"""Monte Carlo test of space-time interaction (Diggle, Chetwynd, Haggkvist and Morris 1995)."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult
from ._rng import random_uniform
from .stk import space_time_k

__all__ = ["space_time_interaction_test"]


def _stat(points, times, s, t, window, tlimits):
    r = space_time_k(points, times, s, t, window=window, tlimits=tlimits)
    return float(sum(sum(float(v) for v in row) for row in r.extra["D"].tolist()))


def space_time_interaction_test(
    points, times, s, t, *, window, tlimits, nsim: int = 99, seed: int = 1
) -> SpatialResult:
    r"""Diggle's Monte Carlo test for space-time clustering.

    With the edge-corrected space-time K estimates of
    :func:`~morie.fn.stk.space_time_k`, the statistic is
    ``T = sum_{s, t} [K_st(s, t) - K_s(s) K_t(t)]``, zero in expectation
    when space and time are independent.  The null distribution comes from
    randomly permuting the event times over the fixed locations, as
    ``splancs::stmctest`` (Diggle et al. 1995); the p-value is
    ``(1 + #{T_sim >= T_obs}) / (nsim + 1)``, large ``T`` indicating events
    close in space tend to be close in time.  Permutations are Fisher-Yates
    shuffles from Philox stream ``k`` of ``seed``, identical in both morie
    arms (``splancs`` uses R's ``sample``).  Pairs count at distance ``<= s``
    and ``<= t`` in every bin; ``splancs`` drops pairs exactly at the largest
    ``s`` or ``t``, so the two differ only when a pair distance ties a top grid
    value.

    :param points: (n, 2) coordinates.
    :param times: Event times (n,).
    :param s: Spatial distances.
    :param t: Temporal distances.
    :param window: ``(xmin, xmax, ymin, ymax)``.
    :param tlimits: ``(tmin, tmax)``.
    :param nsim: Number of permutations.
    :param seed: Philox seed.
    :return: :class:`SpatialResult` with ``statistic``, ``p_value`` and
        ``extra["simulated"]``.

    References
    ----------
    Diggle, P. J., Chetwynd, A. G., Haggkvist, R. and Morris, S. E.
    (1995). Second-order analysis of space-time clustering. *Statistical
    Methods in Medical Research*, 4, 124-136.

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.35, 0.3), (0.8, 0.6), (0.7, 0.15), (0.55, 0.5),
    ...        (0.2, 0.65), (0.9, 0.9), (0.15, 0.95), (0.6, 0.85)]
    >>> tm = [1.0, 7.0, 2.0, 5.0, 3.0, 4.5, 6.0, 9.0, 8.0, 8.5]
    >>> r = space_time_interaction_test(pts, tm, [0.2, 0.4], [2.3, 4.3], window=(0, 1, 0, 1),
    ...                                 tlimits=(0, 10), nsim=19)
    >>> round(r.statistic, 6), r.p_value
    (3.960941, 0.05)
    """
    tm = [float(v) for v in np.asarray(times, dtype=float).tolist()]
    n = len(tm)
    obs = _stat(points, tm, s, t, window, tlimits)
    sims = []
    for k in range(int(nsim)):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=k)]
        p = list(tm)
        for i in range(n - 1, 0, -1):
            j = min(int(u[i] * (i + 1)), i)
            p[i], p[j] = p[j], p[i]
        sims.append(_stat(points, p, s, t, window, tlimits))
    return SpatialResult(
        name="space_time_interaction_test",
        statistic=obs,
        p_value=(1 + sum(1 for v in sims if v >= obs)) / (len(sims) + 1),
        extra={"simulated": sims, "nsim": int(nsim)},
    )


def cheatsheet() -> str:
    return "space_time_interaction_test(points, times, s, t, window, tlimits) -> Diggle space-time Monte Carlo test."
