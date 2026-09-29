# morie.fn -- function file (rootcoder007/morie)
"""SAR Monte-Carlo impact simulation."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from ._rng import random_normal
from .spslx import spatial_impacts


def sarsim(coef, rho, W, nsim=99, vcov=None, seed=0):
    r"""SAR Monte-Carlo impact simulation.

    Draw ``nsim`` parameter vectors ``(rho, beta)`` from their asymptotic
    normal distribution ``N((rho, beta), V)`` (``phi = mean + L z``, ``L L'
    = V`` by Cholesky, ``z`` Philox standard normals), compute the direct,
    indirect and total impacts of every draw
    (:func:`morie.fn.spslx.spatial_impacts`) and summarise them by their
    mean and standard deviation -- the simulation ``spatialreg::impacts(...,
    R = nsim)`` uses for inference on impacts (LeSage and Pace 2009, sec.
    2.7.3).

    Parameters
    ----------
    coef : array-like
        Slopes (no intercept).
    rho : float
        Spatial lag parameter.
    W : array-like, shape (n, n)
        Spatial weights.
    nsim : int
        Number of draws.
    vcov : array-like, shape (k + 1, k + 1)
        Covariance of ``(rho, beta_1..k)`` in that order (required).
    seed : int
        Philox seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean simulated total impact of the first
        covariate; ``extra`` has ``mean`` and ``sd`` dicts (direct,
        indirect, total) and ``point`` (impacts at the estimates).

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press, Boca Raton.

    Examples
    --------
    >>> r = sarsim([2.0], 0.4, [[0, 1, 0], [.5, 0, .5], [0, 1, 0]], nsim=50, vcov=[[0.01, 0.0], [0.0, 0.04]])
    >>> round(r.statistic, 12)
    3.553633906245
    """
    if vcov is None:
        raise ValueError("sarsim needs vcov, the covariance of (rho, beta)")
    b = sd._vec(coef)
    mean = [float(rho)] + b
    k = len(mean)
    L = sd.chol(sd._mat(vcov))
    if len(L) != k:
        raise ValueError("vcov must be (k + 1) x (k + 1) for (rho, beta_1..k)")
    Wm = sd._mat(W)
    z = random_normal(int(nsim) * k, seed=seed)
    z = [float(v) for v in (z.tolist() if hasattr(z, "tolist") else z)]
    keys = ("direct", "indirect", "total")
    draws = {key: [[] for _ in b] for key in keys}
    for s in range(int(nsim)):
        zs = z[s * k : (s + 1) * k]
        phi = [mean[i] + sd.ssum(L[i][m] * zs[m] for m in range(i + 1)) for i in range(k)]
        r = spatial_impacts(phi[0], phi[1:], Wm)
        for key in keys:
            for j in range(len(b)):
                draws[key][j].append(r[key][j])
    mn = {key: [sd.ssum(v) / len(v) for v in draws[key]] for key in keys}
    sdv = {key: [sd.sd(v) for v in draws[key]] for key in keys}
    pt = spatial_impacts(float(rho), b, Wm)
    return SpatialResult(
        name="sarsim",
        statistic=mn["total"][0],
        extra={"mean": mn, "sd": sdv, "point": {key: list(pt[key]) for key in keys}, "nsim": int(nsim)},
    )


sarsim_fn = sarsim


def cheatsheet() -> str:
    return "sarsim(coef, rho, W, nsim, vcov, seed) -> simulated impact means and SDs"
