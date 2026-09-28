# morie.fn -- function file (rootcoder007/morie)
"""Turning bands simulation of isotropic Gaussian random fields in 2-D and 3-D (Matheron 1973; Mantoglou and Wilson 1982)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from ._rng import normal_quantile, random_uniform

__all__ = ["turning_bands"]


def _radial_2d(u, model, a, nu):
    if model == "gaussian":
        return 2.0 / a * math.sqrt(-math.log(u))
    v = 0.5 if model == "exponential" else nu
    return math.sqrt(u ** (-1.0 / v) - 1.0) / a


def _radial_3d_exp(u, a):
    # radial law of the exponential model in 3-D: F(r a) = (2/pi)(atan(r a) - r a / (1 + (r a)^2))
    def F(r):
        return (2.0 / math.pi) * (math.atan(r) - r / (1.0 + r * r))

    lo, hi = 0.0, 1.0
    while F(hi) < u:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if F(mid) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / a


def turning_bands(
    coords,
    cov_model: str = "exponential",
    *,
    sill: float = 1.0,
    range_: float = 1.0,
    nu: float = 0.5,
    n_bands: int = 64,
    n_waves: int = 50,
    angle: float = 0.0,
    ratio: float = 1.0,
    seed: int = 1,
) -> RichResult:
    r"""Gaussian random field by turning bands with spectral line processes.

    The field is ``sqrt(sill / L) sum_l z_l(<x, u_l>)`` over ``L`` bands
    with directions ``u_l`` and independent line processes
    ``z_l(t) = sqrt(2/M) sum_m cos(r_lm t + phi_lm)`` whose frequencies
    ``r_lm`` follow the radial spectral law of the covariance in the
    dimension of the field (Matheron 1973; Mantoglou and Wilson 1982;
    Emery and Lantuejoul 2006).  Averaging ``cos(r <h, u>)`` over the
    directions reproduces ``sill rho(|h|)``; the error vanishes as ``L``
    grows.  2-D: directions ``pi (l + U)/L`` (one random offset); models
    ``exponential``, ``gaussian``, ``matern`` (radial quantiles as
    :func:`~morie.fn.rphase.random_phase_field`), with geometric anisotropy
    (``angle`` of the major axis, ``ratio`` minor/major).  3-D: a
    Fibonacci lattice of directions under a uniform random rotation
    (Shoemake 1992 quaternion from three uniforms of stream 0); ``gaussian`` (``|omega|`` from three
    normals) and ``exponential`` (numerical inversion of
    ``(2/pi)(atan(ra) - ra/(1 + (ra)^2))``).  Philox streams: 0 offset,
    then ``3l + 1`` radii, ``3l + 2`` phases, ``3l + 3`` extra normals.

    :param coords: (n, 2) or (n, 3) locations.
    :param cov_model: ``exponential``, ``gaussian`` or ``matern`` (2-D).
    :param sill: Variance.
    :param range_: Range ``a``.
    :param nu: Matern smoothness.
    :param n_bands: Number of bands ``L``.
    :param n_waves: Waves per band ``M``.
    :param angle: Anisotropy angle (radians, 2-D).
    :param ratio: Anisotropy ratio in ``(0, 1]`` (2-D).
    :param seed: Philox seed.
    :return: :class:`RichResult` with ``field`` and ``directions``.

    References
    ----------
    Matheron, G. (1973). The intrinsic random functions and their
    applications. *Advances in Applied Probability*, 5(3), 439-468.
    Mantoglou, A. and Wilson, J. L. (1982). The turning bands method for
    simulation of random fields using line generation by a spectral method.
    *Water Resources Research*, 18(5), 1379-1394.
    Shoemake, K. (1992). Uniform random rotations. In D. Kirk (ed.),
    *Graphics Gems III*, 124-132. Academic Press, San Diego.
    Emery, X. and Lantuejoul, C. (2006). TBSIM: a computer program for
    conditional simulation of three-dimensional Gaussian random fields via
    the turning bands method. *Computers and Geosciences*, 32(10), 1615-1628.

    Examples
    --------
    >>> r = turning_bands([(0.0, 0.0), (1.0, 0.5)], "gaussian", n_bands=4, n_waves=3, seed=2)
    >>> [round(v, 6) for v in r.field]
    [0.881358, 0.400421]
    """
    X = [tuple(float(v) for v in row) for row in np.asarray(coords, dtype=float).tolist()]
    d = len(X[0])
    if d not in (2, 3):
        raise ValueError("coords must be 2-D or 3-D")
    if cov_model not in ("exponential", "gaussian", "matern") or (d == 3 and cov_model == "matern"):
        raise ValueError("cov_model must be exponential, gaussian or (2-D only) matern")
    if range_ <= 0 or sill < 0 or nu <= 0 or not 0 < ratio <= 1 or n_bands < 1 or n_waves < 1:
        raise ValueError("need range_ > 0, sill >= 0, nu > 0, 0 < ratio <= 1 and positive counts")
    if d == 2 and (angle != 0.0 or ratio != 1.0):
        ca, sa = math.cos(angle), math.sin(angle)
        X = [(ca * x + sa * y, (-sa * x + ca * y) / ratio) for x, y in X]
    L, M = int(n_bands), int(n_waves)
    if d == 2:
        off = float(random_uniform(1, seed=seed, stream=0)[0])
        dirs = [(math.cos(math.pi * (k + off) / L), math.sin(math.pi * (k + off) / L)) for k in range(L)]
    else:
        # Fibonacci lattice under a uniform random rotation (Shoemake 1992 quaternion)
        u1, u2, u3 = (float(v) for v in random_uniform(3, seed=seed, stream=0))
        qx, qy = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2), math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
        qz, qw = math.sqrt(u1) * math.sin(2.0 * math.pi * u3), math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
        Rm = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
        g = math.pi * (3.0 - math.sqrt(5.0))
        dirs = []
        for k in range(L):
            z = 1.0 - (k + 0.5) / L
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            p = (rr * math.cos(g * k), rr * math.sin(g * k), z)
            dirs.append(tuple(sum(Rm[i][j] * p[j] for j in range(3)) for i in range(3)))
    field = [0.0] * len(X)
    c = math.sqrt(2.0 / M)
    for k, u in enumerate(dirs):
        ur = [float(v) for v in random_uniform(M, seed=seed, stream=3 * k + 1)]
        up = [float(v) for v in random_uniform(M, seed=seed, stream=3 * k + 2)]
        if d == 2:
            rad = [_radial_2d(v, cov_model, range_, nu) for v in ur]
        elif cov_model == "gaussian":
            ue = [float(v) for v in random_uniform(2 * M, seed=seed, stream=3 * k + 3)]
            rad = []
            for m in range(M):
                z3 = [
                    float(normal_quantile(ur[m])),
                    float(normal_quantile(ue[2 * m])),
                    float(normal_quantile(ue[2 * m + 1])),
                ]
                rad.append(math.sqrt(2.0) / range_ * math.sqrt(sum(v * v for v in z3)))
        else:
            rad = [_radial_3d_exp(v, range_) for v in ur]
        ph = [2.0 * math.pi * v for v in up]
        for i, x in enumerate(X):
            t = sum(a * b for a, b in zip(x, u))
            field[i] += c * sum(math.cos(rad[m] * t + ph[m]) for m in range(M))
    s = math.sqrt(sill / L)
    return RichResult(payload={"field": [s * v for v in field], "directions": dirs})


turn = turning_bands


def cheatsheet() -> str:
    return "turning_bands(coords, cov_model) -> turning bands Gaussian random field (2-D and 3-D)."


# compact alias per ledger/NAMING.md
turningbands = turning_bands
