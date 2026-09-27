# morie.fn -- function file (rootcoder007/morie)
"""Global Monte Carlo tests of complete spatial randomness: DCLF and MAD on Besag's L-function."""

from __future__ import annotations

import math

from . import _array_core as np
from ._containers import SpatialResult
from ._rng import random_uniform
from .ripk import _rect, isotropic_weight

__all__ = ["csr_global_test"]


def _lfun(P, x0, x1, y0, y1, r):
    n = len(P)
    area = (x1 - x0) * (y1 - y0)
    rmax = r[-1]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                d = math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1])
                if d <= rmax:
                    pairs.append((d, 1.0 / isotropic_weight(P[i][0], P[i][1], d, x0, x1, y0, y1)))
    pairs.sort()
    out, acc, p = [], 0.0, 0
    scale = area / (n * (n - 1.0))
    for h in r:
        while p < len(pairs) and pairs[p][0] <= h:
            acc += pairs[p][1]
            p += 1
        out.append(math.sqrt(scale * acc / math.pi))
    return out


def csr_global_test(
    points,
    window,
    *,
    nsim: int = 99,
    seed: int = 1,
    statistic: str = "dclf",
    rmax: float | None = None,
    nr: int = 513,
) -> SpatialResult:
    r"""Diggle-Cressie-Loosmore-Ford and maximum absolute deviation tests of CSR.

    The discrepancy between Besag's ``L(r) = sqrt(K(r)/pi)`` (isotropic
    K, ``lambda^2 = n(n-1)/|W|^2``) and its CSR value ``r`` over
    ``0 <= r <= rmax`` is either the integrated squared deviation (DCLF;
    Diggle 1986, Cressie 1991, Loosmore and Ford 2006),
    ``u = rmax * mean_r (L(r) - r)^2`` on the grid, or the maximum
    absolute deviation (MAD).  These equal the statistics of
    ``spatstat.explore::dclf.test`` / ``mad.test`` with ``Lest`` and
    ``use.theo = TRUE`` (``nr`` grid points, ``rmax`` from the Kest rule
    ``min(shortside/4, sqrt(1000/(pi lambda)))``).  The null distribution
    comes from ``nsim`` binomial patterns of ``n`` uniform points in the
    window, drawn from Philox stream ``s`` of ``seed`` (identical in both
    morie arms; spatstat uses R's generator); p-value
    ``(1 + #{T_sim >= T_obs}) / (nsim + 1)``.

    :param points: (n, 2) coordinates.
    :param window: ``(xmin, xmax, ymin, ymax)``.
    :param nsim: Number of simulated CSR patterns.
    :param seed: Philox seed.
    :param statistic: ``dclf`` or ``mad``.
    :param rmax: Upper end of the r interval.
    :param nr: Number of r values.
    :return: :class:`SpatialResult` with ``statistic``, ``p_value`` and
        ``extra`` keys ``simulated``, ``r``, ``L``.

    References
    ----------
    Diggle, P. J. (1986). Displaced amacrine cells in the retina of a
    rabbit: analysis of a bivariate spatial point pattern. *Journal of
    Neuroscience Methods*, 18, 115-125.
    Loosmore, N. B. and Ford, E. D. (2006). Statistical inference using the
    G or K point pattern spatial statistics. *Ecology*, 87(8), 1925-1931.

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.35, 0.3), (0.8, 0.6), (0.7, 0.15), (0.55, 0.5),
    ...        (0.2, 0.65), (0.9, 0.9), (0.15, 0.95), (0.6, 0.85)]
    >>> r = csr_global_test(pts, (0, 1, 0, 1), nsim=19, statistic="mad")
    >>> round(r.statistic, 6), r.p_value
    (0.206055, 0.05)
    """
    if statistic not in ("dclf", "mad"):
        raise ValueError("statistic must be 'dclf' or 'mad'")
    P = [(float(a), float(b)) for a, b in np.asarray(points, dtype=float).tolist()]
    n = len(P)
    if n < 2:
        raise ValueError("need at least two points")
    x0, x1, y0, y1 = _rect(window)
    lam = n / ((x1 - x0) * (y1 - y0))
    if rmax is None:
        rmax = min(math.sqrt(1000.0 / (math.pi * lam)), min(x1 - x0, y1 - y0) / 4.0)
    r = [rmax * k / (nr - 1) for k in range(nr)]

    def disc(L):
        dev = [L[k] - r[k] for k in range(nr)]
        if statistic == "mad":
            return max(abs(v) for v in dev)
        return rmax * sum(v * v for v in dev) / nr

    Lobs = _lfun(P, x0, x1, y0, y1, r)
    obs = disc(Lobs)
    sims = []
    for s in range(int(nsim)):
        u = [float(v) for v in random_uniform(2 * n, seed=seed, stream=s)]
        Q = [(x0 + (x1 - x0) * u[2 * i], y0 + (y1 - y0) * u[2 * i + 1]) for i in range(n)]
        sims.append(disc(_lfun(Q, x0, x1, y0, y1, r)))
    return SpatialResult(
        name="csr_global_test",
        statistic=obs,
        p_value=(1 + sum(1 for t in sims if t >= obs)) / (len(sims) + 1),
        extra={"simulated": sims, "nsim": int(nsim), "r": r, "L": Lobs, "method": statistic},
    )


def cheatsheet() -> str:
    return (
        "csr_global_test(points, window) -> DCLF / MAD Monte Carlo test of CSR on L(r) (spatstat dclf.test, mad.test)."
    )
