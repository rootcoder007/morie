# morie.fn -- function file (rootcoder007/morie)
"""Movement and spread processes: lattice and correlated random walks with their mean squared
displacement, planar Brownian motion, the Brownian bridge movement model utilisation distribution,
and site percolation on a grid."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform

__all__ = [
    "lattice_random_walk",
    "correlated_random_walk",
    "crw_msd",
    "planar_brownian_motion",
    "brownian_bridge_ud",
    "site_percolation",
]


def lattice_random_walk(steps: int, *, nwalk: int = 1, seed: int = 1) -> RichResult:
    r"""Simple symmetric random walks on the square lattice ``Z^2`` from the origin.

    Each step moves to one of the four neighbours with probability 1/4
    (Philox stream ``w`` for walk ``w``); the mean squared displacement
    after ``n`` steps is exactly ``n`` (Spitzer 1976). Returns the paths
    and the ensemble mean squared displacement per step.

    References
    ----------
    Spitzer, F. (1976). *Principles of Random Walk*, 2nd edn. Springer.

    Examples
    --------
    >>> r = lattice_random_walk(3, nwalk=2)
    >>> [len(p) for p in r.paths], all(abs(a) + abs(b) <= 3 for p in r.paths for a, b in p)
    ([4, 4], True)
    """
    moves = ((1, 0), (0, 1), (-1, 0), (0, -1))
    paths = []
    for w in range(nwalk):
        u = [float(v) for v in random_uniform(steps, seed=seed, stream=w)]
        x = y = 0
        p = [(0, 0)]
        for t in range(steps):
            dx, dy = moves[min(int(u[t] * 4), 3)]
            x, y = x + dx, y + dy
            p.append((x, y))
        paths.append(p)
    msd = [ssum(p[t][0] ** 2 + p[t][1] ** 2 for p in paths) / nwalk for t in range(steps + 1)]
    return RichResult(payload={"paths": paths, "msd": msd})


def correlated_random_walk(
    steps: int, *, step_length: float = 1.0, kappa: float = 2.0, nwalk: int = 1, seed: int = 1
) -> RichResult:
    r"""Correlated random walks with constant step length and von Mises turning angles (Kareiva and Shigesada 1983).

    The heading changes each step by a von Mises(0, ``kappa``) angle drawn by
    Best and Fisher's (1979) rejection algorithm from Philox uniforms (stream
    ``w``), the initial heading is uniform. Returns the paths, the ensemble
    mean squared displacement and its theoretical value (:func:`crw_msd`).

    References
    ----------
    Kareiva, P. M. and Shigesada, N. (1983). Analyzing insect movement as a
    correlated random walk. *Oecologia*, 56(2-3), 234-238.
    Best, D. J. and Fisher, N. I. (1979). Efficient simulation of the von
    Mises distribution. *Applied Statistics*, 28(2), 152-157.

    Examples
    --------
    >>> r = correlated_random_walk(2, kappa=1e9)
    >>> round(r.msd[2], 6)
    4.0
    """
    tau = 1 + math.sqrt(1 + 4 * kappa * kappa)
    rho = (tau - math.sqrt(2 * tau)) / (2 * kappa)
    rr = (1 + rho * rho) / (2 * rho)
    paths = []
    for w in range(nwalk):
        u = [float(v) for v in random_uniform(1 + 3 * 64 * steps, seed=seed, stream=w)]
        pos = 1
        head = 2 * math.pi * u[0]
        x = y = 0.0
        p = [(0.0, 0.0)]
        for _ in range(steps):
            while True:
                u1, u2, u3 = u[pos], u[pos + 1], u[pos + 2]
                pos += 3
                z = math.cos(math.pi * u1)
                f = (1 + rr * z) / (rr + z)
                c = kappa * (rr - f)
                if c * (2 - c) - u2 > 0 or math.log(c / u2) + 1 - c >= 0:
                    break
            turn = math.copysign(math.acos(max(-1.0, min(1.0, f))), u3 - 0.5)
            head += turn
            x += step_length * math.cos(head)
            y += step_length * math.sin(head)
            p.append((x, y))
        paths.append(p)
    msd = [ssum(q[t][0] ** 2 + q[t][1] ** 2 for q in paths) / nwalk for t in range(steps + 1)]
    return RichResult(
        payload={"paths": paths, "msd": msd, "theory": crw_msd(steps, step_length=step_length, kappa=kappa)}
    )


def _bessel_ratio(kappa):
    # I1(k) / I0(k) by the power series (all terms positive)
    s0 = s1 = 0.0
    t = 1.0
    for m in range(400):
        if m > 0:
            t *= (kappa / 2) ** 2 / (m * m)
        s0 += t
        s1 += t * (kappa / 2) / (m + 1)
        if t < 1e-17 * s0 and m > kappa:
            break
    return s1 / s0 if kappa < 700 else 1 - 1 / (2 * kappa)


def crw_msd(steps: int, *, step_length: float = 1.0, kappa: float = 2.0) -> list:
    r"""Mean squared displacement of a correlated random walk, ``E R_n^2 = n l^2 + 2 l^2 c/(1 - c) (n - (1 - c^n)/(1 - c))`` with ``c = E cos(turn) = I_1(kappa) / I_0(kappa)`` (Kareiva and Shigesada 1983).

    Examples
    --------
    >>> round(crw_msd(1, kappa=2.0)[1], 12)
    1.0
    """
    c = _bessel_ratio(kappa)
    l2 = step_length**2
    out = []
    for n in range(steps + 1):
        out.append(n * l2 + 2 * l2 * c / (1 - c) * (n - (1 - c**n) / (1 - c)) if c < 1 else (n * step_length) ** 2)
    return out


def planar_brownian_motion(n: int, dt: float, *, sigma: float = 1.0, nwalk: int = 1, seed: int = 1) -> RichResult:
    r"""Planar Brownian motion paths ``X_{t+dt} = X_t + sigma sqrt(dt) (Z_1, Z_2)`` from the origin (Philox normals, stream ``w``).

    ``E |X_t|^2 = 2 sigma^2 t``. Returns the paths and the ensemble mean
    squared displacement.

    References
    ----------
    Karatzas, I. and Shreve, S. E. (1991). *Brownian Motion and Stochastic
    Calculus*, 2nd edn. Springer.

    Examples
    --------
    >>> r = planar_brownian_motion(4, 0.5, nwalk=2)
    >>> [len(p) for p in r.paths]
    [5, 5]
    """
    paths = []
    s = sigma * math.sqrt(dt)
    for w in range(nwalk):
        z = [float(v) for v in random_normal(2 * n, seed=seed, stream=w)]
        x = y = 0.0
        p = [(0.0, 0.0)]
        for t in range(n):
            x += s * z[2 * t]
            y += s * z[2 * t + 1]
            p.append((x, y))
        paths.append(p)
    msd = [ssum(q[t][0] ** 2 + q[t][1] ** 2 for q in paths) / nwalk for t in range(n + 1)]
    return RichResult(payload={"paths": paths, "msd": msd})


def brownian_bridge_ud(track, times, xs, ys, *, sig1: float, sig2: float, nalpha: int = 25) -> RichResult:
    r"""Brownian bridge movement model utilisation distribution on a grid (Horne et al. 2007).

    Between consecutive fixes ``a`` and ``b`` (duration ``T``) the position
    at fraction ``alpha`` is normal with mean ``a + alpha (b - a)`` and
    variance ``T alpha (1 - alpha) sig1^2 + (1 - alpha)^2 sig2^2 + alpha^2
    sig2^2`` (motion variance ``sig1^2``, location error ``sig2``); the UD
    at a cell is the duration-weighted time average of these densities,
    ``alpha`` integrated by the midpoint rule with ``nalpha`` nodes, and
    normalised to sum to 1 over the grid cells ``(xs[j], ys[i])``.

    References
    ----------
    Horne, J. S., Garton, E. O., Krone, S. M. and Lewis, J. S. (2007).
    Analyzing animal movements using Brownian bridges. *Ecology*, 88(9),
    2354-2363.

    Examples
    --------
    >>> r = brownian_bridge_ud([(0, 0), (2, 0)], [0, 1], [0, 1, 2], [0], sig1=0.5, sig2=0.1)
    >>> round(sum(sum(row) for row in r.ud), 12), abs(r.ud[0][0] - r.ud[0][2]) < 1e-15
    (1.0, True)
    """
    P = [tuple(float(v) for v in p) for p in track]
    T = [float(v) for v in times]
    X, Y = [float(v) for v in xs], [float(v) for v in ys]
    dens = [[0.0] * len(X) for _ in Y]
    total = T[-1] - T[0]
    for k in range(len(P) - 1):
        dur = T[k + 1] - T[k]
        (ax, ay), (bx, by) = P[k], P[k + 1]
        for m in range(nalpha):
            al = (m + 0.5) / nalpha
            mx, my = ax + al * (bx - ax), ay + al * (by - ay)
            v = dur * al * (1 - al) * sig1**2 + ((1 - al) ** 2 + al**2) * sig2**2
            w = dur / total / nalpha
            for i, y in enumerate(Y):
                for j, x in enumerate(X):
                    dens[i][j] += w * math.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * v)) / (2 * math.pi * v)
    tot = ssum(v for row in dens for v in row)
    return RichResult(payload={"ud": [[v / tot for v in row] for row in dens], "density": dens})


def site_percolation(nrow: int, ncol: int, p: float, *, seed: int = 1, nsim: int = 1) -> RichResult:
    r"""Site percolation on an ``nrow x ncol`` grid: sites open with probability ``p`` (Philox stream ``r``), four-neighbour clusters.

    For each realisation: the cluster labels (union-find), the number of
    clusters, the largest-cluster fraction of all sites, and whether an open
    cluster spans from the left to the right column. The spanning
    probability jumps near ``p_c ~ 0.5927`` on the square lattice
    (Stauffer and Aharony 1994).

    References
    ----------
    Stauffer, D. and Aharony, A. (1994). *Introduction to Percolation
    Theory*, 2nd edn. Taylor and Francis.

    Examples
    --------
    >>> r = site_percolation(3, 3, 1.0)
    >>> r.n_clusters, r.spanning, r.largest_fraction
    ([1], [True], [1.0])
    """
    N = nrow * ncol
    labels, ncl, span, largest = [], [], [], []
    for r in range(nsim):
        u = [float(v) for v in random_uniform(N, seed=seed, stream=r)]
        occ = [v < p for v in u]
        par = list(range(N))

        def find(a, par=par):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        for idx in range(N):
            if not occ[idx]:
                continue
            i, j = divmod(idx, ncol)
            for a, b in ((1, 0), (0, 1)):
                x, y = i + a, j + b
                if x < nrow and y < ncol and occ[x * ncol + y]:
                    ra, rb = find(idx), find(x * ncol + y)
                    if ra != rb:
                        par[max(ra, rb)] = min(ra, rb)
        lab = [find(i) if occ[i] else -1 for i in range(N)]
        sizes = {}
        for v in lab:
            if v >= 0:
                sizes[v] = sizes.get(v, 0) + 1
        left = {lab[i * ncol] for i in range(nrow) if lab[i * ncol] >= 0}
        right = {lab[i * ncol + ncol - 1] for i in range(nrow) if lab[i * ncol + ncol - 1] >= 0}
        labels.append([lab[i * ncol : (i + 1) * ncol] for i in range(nrow)])
        ncl.append(len(sizes))
        span.append(bool(left & right))
        largest.append(max(sizes.values()) / N if sizes else 0.0)
    return RichResult(
        payload={
            "labels": labels,
            "n_clusters": ncl,
            "spanning": span,
            "largest_fraction": largest,
            "spanning_probability": sum(span) / nsim,
        }
    )


def cheatsheet() -> str:
    return (
        "lattice_random_walk / correlated_random_walk / crw_msd / planar_brownian_motion / brownian_bridge_ud / "
        "site_percolation -> movement and spread processes."
    )
