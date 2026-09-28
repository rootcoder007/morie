# morie.fn -- function file (rootcoder007/morie)
"""Movement trajectories: steps, turning angles, sinuosity, displacement, first passage time, random walks, OD matrices."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "trajectory_metrics",
    "mean_squared_displacement",
    "first_passage_time",
    "simulate_walk",
    "od_matrix",
]


def _xy(track):
    P = [(float(a), float(b)) for a, b in np.asarray(track, dtype=float).tolist()]
    if len(P) < 2:
        raise ValueError("a trajectory needs at least two points")
    return P


def trajectory_metrics(track) -> RichResult:
    r"""Step lengths, headings, turning angles, straightness and sinuosity of a planar trajectory.

    Turning angles are heading changes wrapped to ``(-pi, pi]``; the
    straightness index is net displacement over path length (Batschelet
    1981); sinuosity follows Benhamou (2004): ``S = 2 [p ((1 + c)/(1 - c) +
    b^2)]^{-1/2}`` with ``p`` the mean step length, ``c`` the mean cosine of
    the turning angles and ``b`` the coefficient of variation (divisor ``n``)
    of the step lengths.

    References
    ----------
    Benhamou, S. (2004). How to reliably estimate the tortuosity of an
    animal's path: straightness, sinuosity, or fractal dimension? *Journal
    of Theoretical Biology*, 229(2), 209-220.
    Batschelet, E. (1981). *Circular Statistics in Biology*. Academic Press.

    Examples
    --------
    >>> m = trajectory_metrics([(0, 0), (1, 0), (1, 1), (2, 1)])
    >>> m.steps, [round(v, 6) for v in m.turning], round(m.straightness, 6)
    ([1.0, 1.0, 1.0], [1.570796, -1.570796], 0.745356)
    """
    P = _xy(track)
    steps = [math.dist(P[i], P[i + 1]) for i in range(len(P) - 1)]
    head = [math.atan2(P[i + 1][1] - P[i][1], P[i + 1][0] - P[i][0]) for i in range(len(P) - 1)]
    turn = []
    for a, b in zip(head, head[1:]):
        d = b - a
        while d <= -math.pi:
            d += 2 * math.pi
        while d > math.pi:
            d -= 2 * math.pi
        turn.append(d)
    L = ssum(steps)
    net = math.dist(P[0], P[-1])
    p = L / len(steps)
    b = math.sqrt(ssum((s - p) ** 2 for s in steps) / len(steps)) / p
    sinu = float("nan")
    if turn:
        c = ssum(math.cos(t) for t in turn) / len(turn)
        if c < 1:
            sinu = 2.0 / math.sqrt(p * ((1 + c) / (1 - c) + b * b))
    return RichResult(
        payload={
            "steps": steps,
            "headings": head,
            "turning": turn,
            "path_length": L,
            "net_displacement": net,
            "straightness": net / L if L > 0 else float("nan"),
            "sinuosity": sinu,
        }
    )


def mean_squared_displacement(track, max_lag: int | None = None):
    r"""Time-averaged mean squared displacement ``MSD(k) = mean_t ||x_{t+k} - x_t||^2`` for lags ``1..max_lag``.

    Examples
    --------
    >>> mean_squared_displacement([(0, 0), (1, 0), (2, 0), (3, 0)])
    [1.0, 4.0, 9.0]
    """
    P = _xy(track)
    n = len(P)
    K = n - 1 if max_lag is None else min(int(max_lag), n - 1)
    return [
        ssum((P[t + k][0] - P[t][0]) ** 2 + (P[t + k][1] - P[t][1]) ** 2 for t in range(n - k)) / (n - k)
        for k in range(1, K + 1)
    ]


def _exit_time(P, T, i, radius, idx):
    """Interpolated time at which the track, followed along idx from fix i, first leaves the circle about fix i."""
    c = P[i]
    prev = i
    for j in idx:
        if math.dist(P[j], c) > radius:
            d0, d1 = math.dist(P[prev], c), math.dist(P[j], c)
            return T[prev] + (radius - d0) / (d1 - d0) * (T[j] - T[prev])
        prev = j
    return None


def first_passage_time(track, times, radius: float):
    r"""First passage time through a circle of ``radius`` about each fix (Fauchald and Tveraa 2003).

    From each fix the track is followed backwards and forwards until it
    first leaves the circle; exit times are interpolated linearly on the
    crossing segment and the FPT is their difference (``nan`` when the track
    starts or ends inside the circle).

    References
    ----------
    Fauchald, P. and Tveraa, T. (2003). Using first-passage time in the
    analysis of area-restricted search and habitat selection. *Ecology*,
    84(2), 282-288.

    Examples
    --------
    >>> f = first_passage_time([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], [0, 1, 2, 3, 4], 1.5)
    >>> [v if v == v else None for v in f]
    [None, None, 3.0, None, None]
    """
    P = _xy(track)
    T = [float(v) for v in times]
    n = len(P)
    out = []
    for i in range(n):
        fw = _exit_time(P, T, i, radius, range(i + 1, n))
        bw = _exit_time(P, T, i, radius, range(i - 1, -1, -1))
        out.append(fw - bw if fw is not None and bw is not None else float("nan"))
    return out


def simulate_walk(
    n_steps: int,
    *,
    kind: str = "correlated",
    step: float = 1.0,
    rho: float = 0.8,
    mu: float = 2.0,
    target=(0.0, 0.0),
    bias: float = 0.5,
    start=(0.0, 0.0),
    seed: int = 1,
) -> RichResult:
    r"""Random-walk movement models driven by Philox uniforms (streams 0 and 1 of ``seed``).

    - ``correlated``: constant ``step``; turning angles from the wrapped
      Cauchy with concentration ``rho``, ``2 atan(((1 - rho)/(1 + rho)) tan(pi
      (u - 1/2)))`` (Kareiva and Shigesada 1983);
    - ``biased``: the new heading is the circular mean of the wrapped-Cauchy
      correlated heading (weight ``1 - bias``) and the heading to ``target``
      (weight ``bias``) (Benhamou 2006; Codling et al. 2008);
    - ``levy``: uniform headings and Pareto step lengths ``step u^{-1/(mu -
      1)}`` with tail exponent ``mu`` in ``(1, 3]`` (Viswanathan et al. 1999).

    References
    ----------
    Kareiva, P. M. and Shigesada, N. (1983). Analyzing insect movement as a
    correlated random walk. *Oecologia*, 56(2-3), 234-238.
    Codling, E. A., Plank, M. J. and Benhamou, S. (2008). Random walk models
    in biology. *Journal of the Royal Society Interface*, 5(25), 813-834.
    Viswanathan, G. M. et al. (1999). Optimizing the success of random
    searches. *Nature*, 401(6756), 911-914.

    Examples
    --------
    >>> w = simulate_walk(3, kind="correlated", rho=0.0, seed=2)
    >>> [round(v, 6) for v in w.steps]
    [1.0, 1.0, 1.0]
    """
    if kind not in ("correlated", "biased", "levy"):
        raise ValueError("kind must be correlated, biased or levy")
    U1 = [float(v) for v in random_uniform(n_steps, seed=seed, stream=0)]
    U2 = [float(v) for v in random_uniform(n_steps, seed=seed, stream=1)]
    x, y = float(start[0]), float(start[1])
    track = [(x, y)]
    h = 2 * math.pi * U2[0] if kind != "levy" else 0.0
    steps = []
    for k in range(n_steps):
        if kind == "levy":
            h = 2 * math.pi * U2[k]
            s = step * U1[k] ** (-1.0 / (mu - 1.0))
        else:
            turn = 2 * math.atan((1 - rho) / (1 + rho) * math.tan(math.pi * (U1[k] - 0.5)))
            hc = h + turn if k > 0 else h
            if kind == "biased":
                ht = math.atan2(float(target[1]) - y, float(target[0]) - x)
                cx = (1 - bias) * math.cos(hc) + bias * math.cos(ht)
                cy = (1 - bias) * math.sin(hc) + bias * math.sin(ht)
                hc = math.atan2(cy, cx)
            h = hc
            s = step
        x, y = x + s * math.cos(h), y + s * math.sin(h)
        track.append((x, y))
        steps.append(s)
    return RichResult(payload={"track": track, "steps": steps})


def od_matrix(origins, destinations, zones=None, weights=None) -> RichResult:
    r"""Origin-destination matrix: trip counts (or summed weights) by origin zone (rows) and destination zone (columns).

    ``zones`` fixes the order (default the sorted union of labels); also
    returned are the production (row) and attraction (column) totals.

    Examples
    --------
    >>> od_matrix(["a", "a", "b"], ["b", "b", "a"]).matrix
    [[0.0, 2.0], [1.0, 0.0]]
    """
    o, d = list(origins), list(destinations)
    if len(o) != len(d):
        raise ValueError("origins and destinations must have equal length")
    Z = sorted(set(o) | set(d), key=str) if zones is None else list(zones)
    ix = {z: i for i, z in enumerate(Z)}
    w = [1.0] * len(o) if weights is None else [float(v) for v in weights]
    M = [[0.0] * len(Z) for _ in Z]
    for a, b, v in zip(o, d, w):
        M[ix[a]][ix[b]] += v
    return RichResult(
        payload={
            "zones": Z,
            "matrix": M,
            "production": [ssum(r) for r in M],
            "attraction": [ssum(M[i][j] for i in range(len(Z))) for j in range(len(Z))],
        }
    )


def cheatsheet() -> str:
    return "trajectory_metrics / mean_squared_displacement / first_passage_time / simulate_walk -> movement analysis."
