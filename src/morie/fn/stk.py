# morie.fn -- function file (rootcoder007/morie)
"""Space-time K function of Diggle, Chetwynd, Haggkvist and Morris (1995)."""

import math

from . import _array_core as np
from ._containers import DescriptiveResult
from .ripk import isotropic_weight


def space_time_k(
    points,
    times,
    spatial_dists=None,
    temporal_dists=None,
    n_s: int = 10,
    n_t: int = 10,
    *,
    window=None,
    tlimits=None,
) -> DescriptiveResult:
    r"""Spatial, temporal and space-time K functions with edge corrections.

    For events :math:`(x_i, t_i)` in a rectangle :math:`A` and interval
    :math:`[T_0, T_1]` of length :math:`T`, with :math:`w_{ij}` Ripley's
    isotropic factor (the reciprocal of the share of the circle about
    :math:`x_i` through :math:`x_j` inside :math:`A`) and :math:`v_{ij} = 2`
    when :math:`[t_i - u, t_i + u]`, :math:`u = |t_i - t_j|`, reaches either
    time limit (else 1):

    .. math::

        \hat K_s(s) = \frac{|A|}{n(n-1)} \sum_{i \ne j} w_{ij} I(d_{ij} \le s),\quad
        \hat K_t(t) = \frac{T}{n(n-1)} \sum_{i \ne j} v_{ij} I(u_{ij} \le t),

        \hat K_{st}(s, t) = \frac{|A| T}{n(n-1)} \sum_{i \ne j}
        w_{ij} v_{ij} I(d_{ij} \le s) I(u_{ij} \le t).

    Under no space-time interaction :math:`K_{st} = K_s K_t`; the diagnostic
    :math:`D = K_{st} - K_s K_t` and :math:`D_0 = D / (K_s K_t)` are
    returned. These are the estimators of ``splancs::stkhat``.

    :param points: (n, 2) event locations.
    :param times: (n,) event times.
    :param spatial_dists: Distances s (default ``n_s`` values from 0 to half
        the largest inter-event distance).
    :param temporal_dists: Lags t (default ``n_t`` values from 0 to half the
        largest time difference).
    :param n_s: Number of default distances.
    :param n_t: Number of default lags.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)``; defaults to the
        bounding box of the events, which overstates the intensity -- pass
        the study region when known.
    :param tlimits: ``(T0, T1)``; defaults to the range of the times.
    :return: DescriptiveResult; ``value`` is the ``K_st`` matrix (rows
        s, columns t); ``extra`` has ``K_st``, ``K_s``, ``K_t``, ``D``,
        ``D0``, ``spatial_dists``, ``temporal_dists``, ``n``.

    References
    ----------
    Diggle PJ, Chetwynd AG, Haggkvist R, Morris SE (1995). Second-order
    analysis of space-time clustering. Statistical Methods in Medical
    Research 4, 124-136.

    Examples
    --------
    >>> P = [[0.1, 0.2], [0.3, 0.25], [0.8, 0.9], [0.5, 0.5]]
    >>> r = space_time_k(P, [1.0, 1.5, 4.0, 2.0], [0.3], [1.0], window=(0, 1, 0, 1), tlimits=(0, 5))
    >>> round(float(r.extra["K_s"][0]), 9), round(float(r.extra["K_t"][0]), 9)
    (0.226216471, 2.916666667)
    """
    P = [[float(v) for v in p] for p in np.asarray(points, dtype=float).tolist()]
    T = [float(v) for v in np.asarray(times, dtype=float).ravel().tolist()]
    n = len(P)
    if n < 2 or len(T) != n:
        raise ValueError("need at least two events with one time each")
    if window is None:
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        window = (min(xs), max(xs), min(ys), max(ys))
    x0, x1, y0, y1 = (float(v) for v in window)
    t0, t1 = (float(v) for v in (tlimits if tlimits is not None else (min(T), max(T))))
    if spatial_dists is None:
        dmax = max(math.dist(a, b) for a in P for b in P)
        spatial_dists = [dmax / 2 * k / (n_s - 1) for k in range(n_s)] if n_s > 1 else [dmax / 2]
    if temporal_dists is None:
        umax = max(T) - min(T)
        temporal_dists = [umax / 2 * k / (n_t - 1) for k in range(n_t)] if n_t > 1 else [umax / 2]
    s = [float(v) for v in np.asarray(spatial_dists, dtype=float).ravel().tolist()]
    t = [float(v) for v in np.asarray(temporal_dists, dtype=float).ravel().tolist()]
    area, span = (x1 - x0) * (y1 - y0), t1 - t0
    ks = [0.0] * len(s)
    kt = [0.0] * len(t)
    kst = [[0.0] * len(t) for _ in s]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.dist(P[i], P[j])
            u = abs(T[i] - T[j])
            w = 1.0 / isotropic_weight(P[i][0], P[i][1], d, x0, x1, y0, y1)
            v = 2.0 if (T[i] - t0 <= u or t1 - T[i] <= u) else 1.0
            for a, sa in enumerate(s):
                if d <= sa:
                    ks[a] += w
            for b, tb in enumerate(t):
                if u <= tb:
                    kt[b] += v
                    for a, sa in enumerate(s):
                        if d <= sa:
                            kst[a][b] += w * v
    c = n * (n - 1)
    ks = [area * k / c for k in ks]
    kt = [span * k / c for k in kt]
    kst = [[area * span * k / c for k in row] for row in kst]
    D = [[kst[a][b] - ks[a] * kt[b] for b in range(len(t))] for a in range(len(s))]
    D0 = [
        [D[a][b] / (ks[a] * kt[b]) if ks[a] * kt[b] > 0 else float("nan") for b in range(len(t))] for a in range(len(s))
    ]
    K = np.asarray(kst, dtype=float)
    return DescriptiveResult(
        name="space_time_k",
        value=K,
        extra={
            "K_st": K,
            "K_s": np.asarray(ks, dtype=float),
            "K_t": np.asarray(kt, dtype=float),
            "D": np.asarray(D, dtype=float),
            "D0": np.asarray(D0, dtype=float),
            "spatial_dists": np.asarray(s, dtype=float),
            "temporal_dists": np.asarray(t, dtype=float),
            "n": n,
        },
    )


stk = space_time_k


def cheatsheet() -> str:
    return "space_time_k({}) -> Diggle et al. (1995) space-time K functions with edge corrections."


# compact alias per ledger/NAMING.md
spacetimek = space_time_k
