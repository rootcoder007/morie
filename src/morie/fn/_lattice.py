# morie.fn -- function file (rootcoder007/morie)
"""Shared kernels of the lattice autocorrelation front ends (``lac*``), in ``spdep`` conventions.

Global Moran's I, Geary's C and Getis-Ord G with their analytic moments,
local Moran's I with the conditional randomisation variance, conditional
permutation (Philox) inference, Lee's L, the Moran scatterplot quadrants
and the empirical semivariogram. R twin: ``R/LatticeAutocorrelation.R``
(helpers ``.lat_*``).
"""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._rng import random_uniform
from ._rrng_core import pnorm

__all__: list = []


def mat(A):
    A = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in r] for r in A]


def vec(v):
    v = v.tolist() if hasattr(v, "tolist") else v
    return [float(t) for t in v]


def mv(W, v):
    return [ssum(a * b for a, b in zip(r, v)) for r in W]


def upper(z):
    return 1.0 - float(pnorm(z))


def constants(W):
    """``n, S0, S1, S2`` (Cliff and Ord 1981)."""
    n = len(W)
    S0 = ssum(v for r in W for v in r)
    S1 = 0.5 * ssum((W[i][j] + W[j][i]) ** 2 for i in range(n) for j in range(n))
    S2 = ssum((ssum(W[i]) + ssum(W[j][i] for j in range(n))) ** 2 for i in range(n))
    return n, S0, S1, S2


def centre(x):
    m = ssum(x) / len(x)
    return [v - m for v in x]


def moran(x, W):
    """Moran's I with its randomisation moments (``spdep::moran.test``)."""
    n, S0, S1, S2 = constants(W)
    z = centre(x)
    zz = ssum(v * v for v in z)
    moran_i = n / S0 * ssum(a * b for a, b in zip(z, mv(W, z))) / zz
    K = n * ssum(v**4 for v in z) / zz**2
    E = -1.0 / (n - 1)
    nn = n * n
    VI = n * (S1 * (nn - 3 * n + 3) - n * S2 + 3 * S0 * S0)
    VI -= K * (S1 * (nn - n) - 2 * n * S2 + 6 * S0 * S0)
    VI = VI / ((n - 1) * (n - 2) * (n - 3) * S0 * S0) - E * E
    zs = (moran_i - E) / math.sqrt(VI)
    return {"statistic": moran_i, "expected": E, "variance": VI, "z": zs, "p_value": upper(zs)}


def geary_c(x, W):
    n, S0, _S1, _S2 = constants(W)
    z = centre(x)
    zz = ssum(v * v for v in z)
    num = ssum(W[i][j] * (x[i] - x[j]) ** 2 for i in range(n) for j in range(n))
    return (n - 1) / (2 * S0) * num / zz, n * ssum(v**4 for v in z) / zz**2


def geary(x, W, randomisation=True):
    """Geary's C with its moments (``spdep::geary.test``); ``p`` for positive autocorrelation."""
    n, S0, S1, S2 = constants(W)
    C, K = geary_c(x, W)
    n1, n2, n3, nn, S02 = n - 1, n - 2, n - 3, n * n, S0 * S0
    if randomisation:
        V = n1 * S1 * (nn - 3 * n + 3 - K * n1)
        V -= 0.25 * (n1 * S2 * (nn + 3 * n - 6 - K * (nn - n + 2)))
        V += S02 * (nn - 3 - K * n1 * n1)
        V /= n * n2 * n3 * S02
    else:
        V = ((2 * S1 + S2) * n1 - 4 * S02) / (2 * (n + 1) * S02)
    zs = (1.0 - C) / math.sqrt(V)
    return {"statistic": C, "expected": 1.0, "variance": V, "z": zs, "p_value": upper(zs), "K": K}


def getis_ord(x, W):
    """Global G with its moments (``spdep::globalG.test``, ``B1correct = TRUE``)."""
    n, S0, S1, S2 = constants(W)
    if min(x) < 0:
        raise ValueError("Getis-Ord G needs non-negative x")
    num = ssum(a * b for a, b in zip(x, mv(W, x)))
    sx = ssum(x)
    sx2, sx3, sx4 = ssum(v * v for v in x), ssum(v**3 for v in x), ssum(v**4 for v in x)
    G = num / (sx * sx - sx2)
    EG = S0 / (n * (n - 1))
    nn, S02 = n * n, S0 * S0
    B0 = (nn - 3 * n + 3) * S1 - n * S2 + 3 * S02
    B1 = -((nn - n) * S1 - 2 * n * S2 + 6 * S02)
    B2 = -(2 * n * S1 - (n + 3) * S2 + 6 * S02)
    B3 = 4 * (n - 1) * S1 - 2 * (n + 1) * S2 + 8 * S02
    B4 = S1 - S2 + S02
    VG = (B0 * sx2**2 + B1 * sx4 + B2 * sx * sx * sx2 + B3 * sx * sx3 + B4 * sx**4) / (
        (sx * sx - sx2) ** 2 * n * (n - 1) * (n - 2) * (n - 3)
    ) - EG * EG
    zs = (G - EG) / math.sqrt(VG)
    return {"statistic": G, "expected": EG, "variance": VG, "z": zs, "p_value": upper(zs)}


def local_moran(x, W):
    """Local Moran's I_i with the conditional randomisation moments (``spdep::localmoran`` defaults)."""
    n = len(W)
    z = centre(x)
    m2 = ssum(v * v for v in z) / n
    lz = mv(W, z)
    Ii = [a / m2 * b for a, b in zip(z, lz)]
    Wi = [ssum(r) for r in W]
    Wi2 = [ssum(v * v for v in r) for r in W]
    E = [-(z[i] ** 2 * Wi[i]) / ((n - 1) * m2) for i in range(n)]
    V = [
        (z[i] / m2) ** 2 * (n / (n - 2)) * (Wi2[i] - Wi[i] ** 2 / (n - 1)) * (m2 - z[i] ** 2 / (n - 1))
        for i in range(n)
    ]
    Z = [(a - b) / math.sqrt(c) if c > 0 else math.nan for a, b, c in zip(Ii, E, V)]
    P = [2.0 * upper(abs(t)) if t == t else math.nan for t in Z]
    return {"Ii": Ii, "expected": E, "variance": V, "z": Z, "p_value": P, "zx": z, "lag": lz}


def quadrants(x, W):
    """Moran-scatterplot quadrant of each unit from ``z = x - mean`` and ``Wz`` (1 HH, 2 LH, 3 LL, 4 HL)."""
    z = centre(x)
    lz = mv(W, z)
    q = []
    for a, b in zip(z, lz):
        if a > 0:
            q.append(1 if b > 0 else 4)
        else:
            q.append(2 if b > 0 else 3)
    return q, z, lz


def perm_order(u):
    return sorted(range(len(u)), key=lambda k: u[k])


def local_moran_perm(x, W, nsim, seed):
    """Conditional permutation pseudo p-values of local Moran's I (Anselin 1995).

    For unit ``i`` the other ``n - 1`` centred values are permuted
    (the order of Philox uniforms, block ``s * n + i``) and
    ``I_i* = z_i / m2 * sum_j w_ij z*_j`` recomputed; the folded pseudo
    p-value is ``(1 + min(#{I* >= I_i}, #{I* <= I_i})) / (nsim + 1)``.
    """
    n = len(W)
    z = centre(x)
    m2 = ssum(v * v for v in z) / n
    Ii = [a / m2 * b for a, b in zip(z, mv(W, z))]
    u = random_uniform(nsim * n * (n - 1), seed=seed)
    u = [float(v) for v in (u.tolist() if hasattr(u, "tolist") else u)]
    p, mean_sim, sd_sim = [], [], []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        wi = [W[i][j] for j in others]
        sims = []
        for s in range(nsim):
            blk = u[(s * n + i) * (n - 1) : (s * n + i + 1) * (n - 1)]
            order = perm_order(blk)
            sims.append(z[i] / m2 * ssum(wi[k] * z[others[order[k]]] for k in range(n - 1)))
        ge = sum(1 for v in sims if v >= Ii[i])
        le = sum(1 for v in sims if v <= Ii[i])
        p.append((1 + min(ge, le)) / (nsim + 1))
        m = ssum(sims) / nsim
        mean_sim.append(m)
        sd_sim.append(math.sqrt(ssum((v - m) ** 2 for v in sims) / (nsim - 1)))
    return Ii, p, mean_sim, sd_sim


def geary_perm(x, W, nsim, seed):
    """Permutation test of Geary's C: ``p = (1 + #{C* <= C}) / (nsim + 1)`` (positive autocorrelation)."""
    n = len(W)
    C, _ = geary_c(x, W)
    u = random_uniform(nsim * n, seed=seed)
    u = [float(v) for v in (u.tolist() if hasattr(u, "tolist") else u)]
    sims = []
    for s in range(nsim):
        order = perm_order(u[s * n : (s + 1) * n])
        sims.append(geary_c([x[k] for k in order], W)[0])
    k = sum(1 for v in sims if v <= C)
    return C, (1 + k) / (nsim + 1), sims


def lee_l(x, y, W):
    """Lee's (2001) bivariate L and its local components (``spdep::lee``)."""
    n = len(W)
    zx, zy = centre(x), centre(y)
    sx, sy = math.sqrt(ssum(v * v for v in zx)), math.sqrt(ssum(v * v for v in zy))
    lx, ly = mv(W, zx), mv(W, zy)
    S2 = ssum(ssum(r) ** 2 for r in W)
    L = n / S2 * ssum(a * b for a, b in zip(lx, ly)) / (sx * sy)
    return L, [n * a * b / (sx * sy) for a, b in zip(lx, ly)]


def semivariogram(z, coords, n_lags=15, cutoff=None):
    """Matheron's estimator ``gamma(h) = sum (z_i - z_j)^2 / (2 N(h))`` in equal-width distance bins.

    gstat defaults: ``cutoff`` one third of the bounding-box diagonal,
    ``n_lags = 15`` bins of width ``cutoff / 15``, bin ``k`` holding pairs
    with ``k w < d <= (k + 1) w``; ``dist`` is the mean pair distance.
    """
    n = len(z)
    if cutoff is None:
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        cutoff = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 3.0
    w = cutoff / n_lags
    np_ = [0] * n_lags
    ds = [0.0] * n_lags
    gs = [0.0] * n_lags
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            if d > cutoff or d == 0.0:
                continue
            k = min(int(math.ceil(d / w)) - 1, n_lags - 1)
            np_[k] += 1
            ds[k] += d
            gs[k] += (z[i] - z[j]) ** 2
    keep = [k for k in range(n_lags) if np_[k] > 0]
    return {
        "np": [np_[k] for k in keep],
        "dist": [ds[k] / np_[k] for k in keep],
        "gamma": [gs[k] / (2 * np_[k]) for k in keep],
        "cutoff": cutoff,
        "width": w,
    }
