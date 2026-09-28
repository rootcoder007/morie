"""Spectral (FFT) Gaussian random field simulation by circulant embedding."""

from __future__ import annotations

import math

from . import _array_core as np
from ._containers import SpatialResult
from ._rng import random_normal
from ._sci_core import kv

_MODELS = ("exponential", "gaussian", "matern", "spherical")


def _corr(h, model, nu):
    if h <= 0.0:
        return 1.0
    if model == "exponential":
        return math.exp(-h)
    if model == "gaussian":
        return math.exp(-h * h)
    if model == "spherical":
        return 1.0 - 1.5 * h + 0.5 * h**3 if h < 1.0 else 0.0
    return 2.0 ** (1.0 - nu) / math.gamma(nu) * h**nu * float(kv(nu, h))


def spectral_grf_sim(
    coords: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_sims: int = 1,
    seed: int = 42,
) -> SpatialResult:
    r"""Simulate a stationary Gaussian random field by circulant embedding.

    The covariance ``C(h) = sill * rho(|A h| / range)`` (plus a nugget) on a
    regular grid is embedded in a torus of ``g`` times the grid size whose
    base block holds ``C`` at the wrapped lags ``min(i, N - i)``; its
    eigenvalues are the 2-D FFT of that block (Wood and Chan 1994; Dietrich
    and Newsam 1997).  The real part of ``FFT(sqrt(S / N) (e1 + i e2))``, with
    independent standard normal ``e1, e2``, then has covariance exactly
    ``C`` on the grid.  ``g`` doubles (2, 4, 8) until no eigenvalue is below
    ``-1e-8`` of the largest; otherwise negatives are clipped and
    ``embedding_exact`` is False.

    Correlation models ``rho``: ``exponential`` ``exp(-h)``, ``gaussian``
    ``exp(-h^2)``, ``spherical`` ``1 - 1.5h + 0.5h^3`` (``h < 1``) and
    ``matern`` ``2^(1-nu)/Gamma(nu) h^nu K_nu(h)`` (geoR / fields
    parameterisation).  Geometric anisotropy rotates lags by ``angle``
    (radians, major axis from the x axis) and divides the minor-axis
    component by ``ratio <= 1``.  Normals come from Philox (``e1``, ``e2``
    of realisation ``k`` are streams ``2k`` and ``2k + 1`` of ``seed``,
    row-major over the torus; the nugget uses stream ``2 n_sims + k``), so
    the R arm gives the same fields.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Points of a regular grid.
    cov_model : str
        ``exponential``, ``gaussian``, ``matern`` or ``spherical``.
    cov_params : dict, optional
        ``sill`` (1), ``range`` (1), ``nugget`` (0), ``nu`` (0.5, Matern),
        ``angle`` (0) and ``ratio`` (1).
    n_sims : int
        Number of realisations.
    seed : int
        Philox seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean of the first realisation; ``extra`` has
        ``simulations`` (n_sims, n), ``eigenvalues`` of the embedding,
        ``torus`` (Nx, Ny), ``embedding_exact`` and ``min_eigenvalue_ratio``.

    References
    ----------
    Wood, A. T. A. and Chan, G. (1994). Simulation of stationary Gaussian
    processes in [0, 1]^d. *Journal of Computational and Graphical
    Statistics*, 3(4), 409-432.
    Dietrich, C. R. and Newsam, G. N. (1997). Fast and exact simulation of
    stationary Gaussian processes through circulant embedding of the
    covariance matrix. *SIAM Journal on Scientific Computing*, 18(4),
    1088-1107.

    Examples
    --------
    >>> g = [(x * 0.5, y * 0.5) for x in range(4) for y in range(3)]
    >>> r = spectral_grf_sim(g, "spherical", {"range": 1.0}, seed=1)
    >>> r.extra["torus"], r.extra["embedding_exact"]
    ((8, 6), True)
    """
    coords = [(float(a), float(b)) for a, b in np.asarray(coords, dtype=float).tolist()]
    n = len(coords)
    params = dict(cov_params or {})
    sill = float(params.get("sill", 1.0))
    r = float(params.get("range", 1.0))
    nug = float(params.get("nugget", 0.0))
    nu = float(params.get("nu", 0.5))
    ang = float(params.get("angle", 0.0))
    ratio = float(params.get("ratio", 1.0))
    if cov_model not in _MODELS:
        raise ValueError(f"cov_model must be one of {_MODELS}")
    if r <= 0 or sill < 0 or nug < 0 or nu <= 0 or not 0 < ratio <= 1:
        raise ValueError("need range > 0, sill >= 0, nugget >= 0, nu > 0, 0 < ratio <= 1")
    cx = [c[0] for c in coords]
    cy = [c[1] for c in coords]
    ux = sorted(set(round(v, 8) for v in cx))
    uy = sorted(set(round(v, 8) for v in cy))
    nx, ny = len(ux), len(uy)
    stx = ux[1] - ux[0] if nx > 1 else 1.0
    sty = uy[1] - uy[0] if ny > 1 else 1.0
    for u, st in ((ux, stx), (uy, sty)):
        for a, b in zip(u, u[1:]):
            if abs((b - a) - st) > 1e-6 * max(1.0, abs(st)):
                raise ValueError("coords must lie on a regular grid")
    ix = [int(round((round(v, 8) - ux[0]) / stx)) for v in cx]
    iy = [int(round((round(v, 8) - uy[0]) / sty)) for v in cy]
    ca, sa = math.cos(ang), math.sin(ang)

    def cov(dx, dy):
        u = ca * dx + sa * dy
        v = (-sa * dx + ca * dy) / ratio
        return sill * _corr(math.hypot(u, v) / r, cov_model, nu)

    for grow in (2, 4, 8):
        Nx, Ny = grow * nx, grow * ny
        # wrapped signed lags keep anisotropic covariances symmetric on the torus
        C_embed = [
            [cov((i if i <= Nx // 2 else i - Nx) * stx, (j if j <= Ny // 2 else j - Ny) * sty) for j in range(Ny)]
            for i in range(Nx)
        ]
        S = [[float(v.real) for v in row] for row in np.fft.fft2(np.array(C_embed)).tolist()]
        smax = max(max(row) for row in S)
        smin = min(min(row) for row in S)
        exact = smin >= -1e-8 * smax
        if exact:
            break
    N = Nx * Ny
    amp = [[math.sqrt(max(v, 0.0) / N) for v in row] for row in S]
    sims = []
    for k in range(int(n_sims)):
        e1 = [float(v) for v in random_normal(N, seed=seed, stream=2 * k)]
        e2 = [float(v) for v in random_normal(N, seed=seed, stream=2 * k + 1)]
        w = [[amp[i][j] * complex(e1[i * Ny + j], e2[i * Ny + j]) for j in range(Ny)] for i in range(Nx)]
        f = np.fft.fft2(np.array(w)).tolist()
        vals = [f[ix[m]][iy[m]].real for m in range(n)]
        if nug > 0:
            z = [float(v) for v in random_normal(n, seed=seed, stream=2 * int(n_sims) + k)]
            vals = [a + math.sqrt(nug) * b for a, b in zip(vals, z)]
        sims.append(vals)
    return SpatialResult(
        name="spectral_grf_sim",
        statistic=sum(sims[0]) / n if sims else float("nan"),
        p_value=None,
        extra={
            "simulations": np.asarray(sims, dtype=float) if sims else np.zeros((0, n)),
            "eigenvalues": S,
            "torus": (Nx, Ny),
            "embedding_exact": bool(exact),
            "min_eigenvalue_ratio": float(smin / smax),
        },
    )


sgsps = spectral_grf_sim


def cheatsheet() -> str:
    return "spectral_grf_sim(coords, cov_model, cov_params) -> circulant-embedding Gaussian random field (Wood-Chan)."
