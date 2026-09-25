"""Spectral (FFT-based) Gaussian random field simulation."""

from __future__ import annotations

from . import _array_core as np

from ._containers import SpatialResult


def spectral_grf_sim(
    coords: np.ndarray,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_sims: int = 1,
    seed: int = 42,
) -> SpatialResult:
    r"""Simulate a GRF using the spectral (Fourier) method.

    Generates the field on a regular grid via circulant embedding
    and FFT.

    Parameters
    ----------
    coords : np.ndarray
        Grid coordinates, shape ``(n, 2)``. Must be on a regular grid.
    cov_model : str
        ``"exponential"`` or ``"gaussian"``.
    cov_params : dict, optional
        ``{"sill", "range", "nugget"}``.
    n_sims : int
        Number of realizations.
    seed : int
        RNG seed.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean of first realization.
        ``extra`` has ``simulations``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 7.

    .. epigraph::

    """
    import math

    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    params = cov_params or {"sill": 1.0, "range": 1.0, "nugget": 0.0}
    sill = float(params.get("sill", 1.0))
    r = float(params.get("range", 1.0))
    nug = float(params.get("nugget", 0.0))
    if cov_model not in ("exponential", "gaussian"):
        raise ValueError("cov_model must be 'exponential' or 'gaussian'")
    if r <= 0 or sill < 0 or nug < 0:
        raise ValueError("need range > 0, sill >= 0, nugget >= 0")
    cx = [float(v) for v in coords[:, 0].tolist()]
    cy = [float(v) for v in coords[:, 1].tolist()]
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

    def cov(h):
        return sill * (math.exp(-(h / r) ** 2) if cov_model == "gaussian"
                       else math.exp(-h / r))

    # Circulant embedding (Wood and Chan 1994; Dietrich and Newsam 1997):
    # the base block holds C at the WRAPPED lag min(i, N - i), so the
    # torus covariance restricted to the nx x ny corner is exactly C.
    # The old code put C in one corner and zeros elsewhere, which is not
    # a circulant covariance at all, and scaled the field by 1/N.
    # A torus that is too small can leave negative eigenvalues; the
    # remedy (Wood and Chan 1994) is to enlarge it, up to 8x here. Tiny
    # negatives are round-off; if large ones remain they are clipped and
    # ``embedding_exact`` is False.
    for grow in (2, 4, 8):
        Nx, Ny = grow * nx, grow * ny
        C_embed = [[cov(math.hypot(min(i, Nx - i) * stx,
                                   min(j, Ny - j) * sty))
                    for j in range(Ny)] for i in range(Nx)]
        S = [[float(v.real) for v in row]
             for row in np.fft.fft2(np.array(C_embed)).tolist()]
        smax = max(max(row) for row in S)
        smin = min(min(row) for row in S)
        exact = smin >= -1e-8 * smax
        if exact:
            break
    N = Nx * Ny
    amp = [[math.sqrt(max(v, 0.0) / N) for v in row] for row in S]
    sims = np.empty((n_sims, n))
    for k in range(n_sims):
        e1 = rng.standard_normal((Nx, Ny)).tolist()
        e2 = rng.standard_normal((Nx, Ny)).tolist()
        w = [[amp[i][j] * complex(e1[i][j], e2[i][j]) for j in range(Ny)]
             for i in range(Nx)]
        # the real part of FFT(sqrt(S/N) eps) has covariance exactly C
        f = np.fft.fft2(np.array(w)).tolist()
        vals = [f[ix[m]][iy[m]].real for m in range(n)]
        if nug > 0:
            z = rng.standard_normal(n).tolist()
            vals = [v + math.sqrt(nug) * t for v, t in zip(vals, z)]
        sims[k] = vals
    return SpatialResult(
        name="spectral_grf_sim",
        statistic=float(np.mean(sims[0])),
        p_value=None,
        extra={"simulations": sims, "embedding_exact": bool(exact),
               "min_eigenvalue_ratio": float(smin / smax)},
    )


sgsps = spectral_grf_sim


def cheatsheet() -> str:
    return "spectral_grf_sim({}) -> Spectral (FFT-based) Gaussian random field simulation."


# compact alias per ledger/NAMING.md
spectralgrfsim = spectral_grf_sim
