# morie.fn -- function file (rootcoder007/morie)
"""Random-phase spectral (Shinozuka) simulation of isotropic Gaussian random fields."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["random_phase_field"]


def random_phase_field(
    coords,
    cov_model: str = "exponential",
    *,
    sill: float = 1.0,
    range_: float = 1.0,
    nu: float = 0.5,
    n_waves: int = 500,
    seed: int = 1,
) -> RichResult:
    r"""Isotropic Gaussian random field by the random-phase spectral method.

    ``f(s) = sqrt(2 sill / M) sum_m cos(omega_m . s + phi_m)`` with
    ``phi_m ~ U(0, 2 pi)`` and wave vectors ``omega_m`` drawn from the
    normalised spectral density of the covariance (Shinozuka 1971;
    Shinozuka and Deodatis 1991); since ``E cos(omega . h) = rho(h)`` the
    covariance is exactly ``sill rho(h)`` and the field tends to Gaussian as
    ``M`` grows.  In two dimensions the direction is uniform and the radius
    ``|omega|`` has a closed-form quantile:

    - ``gaussian`` ``rho = exp(-(h/a)^2)``: ``|omega| = (2/a) sqrt(-log U)``;
    - ``matern`` (smoothness ``nu``, ``exponential`` = ``nu = 1/2``):
      spectral density proportional to ``(1 + a^2 |omega|^2)^-(nu + 1)``, so
      ``|omega| = sqrt(U^(-1/nu) - 1) / a``.

    Uniforms come from Philox streams 0 (radius), 1 (direction), 2 (phase)
    of ``seed``, identical in both morie arms.

    :param coords: (n, 2) locations (need not be a grid).
    :param cov_model: ``exponential``, ``gaussian`` or ``matern``.
    :param sill: Variance.
    :param range_: Range ``a``.
    :param nu: Matern smoothness.
    :param n_waves: Number of cosine waves ``M``.
    :param seed: Philox seed.
    :return: :class:`RichResult` with ``field``, ``omega`` (M pairs),
        ``phase``.

    References
    ----------
    Shinozuka, M. (1971). Simulation of multivariate and multidimensional
    random processes. *Journal of the Acoustical Society of America*,
    49(1B), 357-367.
    Shinozuka, M. and Deodatis, G. (1991). Simulation of stochastic
    processes by spectral representation. *Applied Mechanics Reviews*,
    44(4), 191-204.

    Examples
    --------
    >>> r = random_phase_field([(0.0, 0.0), (1.0, 0.5)], "gaussian", n_waves=4, seed=2)
    >>> [round(v, 6) for v in r.field]
    [0.481186, -0.249627]
    """
    if cov_model not in ("exponential", "gaussian", "matern"):
        raise ValueError("cov_model must be exponential, gaussian or matern")
    if range_ <= 0 or sill < 0 or nu <= 0 or n_waves < 1:
        raise ValueError("need range_ > 0, sill >= 0, nu > 0, n_waves >= 1")
    P = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    M = int(n_waves)
    ur = [float(v) for v in random_uniform(M, seed=seed, stream=0)]
    ut = [float(v) for v in random_uniform(M, seed=seed, stream=1)]
    up = [float(v) for v in random_uniform(M, seed=seed, stream=2)]
    v = 0.5 if cov_model == "exponential" else float(nu)
    om, ph = [], []
    for m in range(M):
        u = ur[m]
        if cov_model == "gaussian":
            rad = 2.0 / range_ * math.sqrt(-math.log(u))
        else:
            rad = math.sqrt(u ** (-1.0 / v) - 1.0) / range_
        th = 2.0 * math.pi * ut[m]
        om.append((rad * math.cos(th), rad * math.sin(th)))
        ph.append(2.0 * math.pi * up[m])
    c = math.sqrt(2.0 * sill / M)
    field = [c * sum(math.cos(om[m][0] * x + om[m][1] * y + ph[m]) for m in range(M)) for x, y in P]
    return RichResult(payload={"field": field, "omega": om, "phase": ph})


def cheatsheet() -> str:
    return "random_phase_field(coords, cov_model) -> Shinozuka random-phase spectral Gaussian field."
