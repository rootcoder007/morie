"""Turning bands simulation (SpatialResult wrapper of :func:`morie.fn.zstbs.turning_bands`)."""

from __future__ import annotations

from ._containers import SpatialResult
from .zstbs import turning_bands


def turning_bands_sim(
    coords,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    n_bands: int = 100,
    seed: int = 42,
    n_waves: int = 50,
) -> SpatialResult:
    r"""Simulate a Gaussian random field by the turning bands method.

    Delegates to :func:`morie.fn.zstbs.turning_bands` (spectral line
    processes on ``n_bands`` bands; Mantoglou and Wilson 1982) with
    ``sill`` and ``range`` from ``cov_params``; a ``nugget`` adds independent
    normal noise of that variance (Philox stream ``10**6``).

    Parameters
    ----------
    coords : array-like
        Simulation coordinates, shape ``(n, 2)`` or ``(n, 3)``.
    cov_model : str
        ``"exponential"``, ``"gaussian"`` or ``"matern"`` (2-D; ``nu`` in
        ``cov_params``).
    cov_params : dict, optional
        ``{"sill", "range", "nugget", "nu"}``.
    n_bands : int
        Number of bands.
    seed : int
        Philox seed.
    n_waves : int
        Waves per band.

    Returns
    -------
    SpatialResult
        ``statistic`` is the mean of the simulated field; ``extra`` has
        ``simulated_values`` and ``n_bands``.

    References
    ----------
    Mantoglou, A. and Wilson, J. L. (1982). The turning bands method for
    simulation of random fields using line generation by a spectral method.
    *Water Resources Research*, 18(5), 1379-1394.

    Examples
    --------
    >>> r = turning_bands_sim([(0.0, 0.0), (1.0, 0.5)], "gaussian", n_bands=4, seed=2, n_waves=3)
    >>> [round(v, 6) for v in r.extra["simulated_values"]]
    [0.881358, 0.400421]
    """
    p = cov_params or {}
    z = list(
        turning_bands(
            coords,
            cov_model,
            sill=float(p.get("sill", 1.0)),
            range_=float(p.get("range", 1.0)),
            nu=float(p.get("nu", 0.5)),
            n_bands=n_bands,
            n_waves=n_waves,
            seed=seed,
        ).field
    )
    nug = float(p.get("nugget", 0.0))
    if nug > 0:
        from ._rng import random_normal

        e = random_normal(len(z), seed=seed, stream=10**6)
        z = [v + nug**0.5 * float(w) for v, w in zip(z, e)]
    return SpatialResult(
        name="turning_bands_sim",
        statistic=sum(z) / len(z),
        p_value=None,
        extra={"simulated_values": z, "n_bands": n_bands},
    )


sgtbn = turning_bands_sim


def cheatsheet() -> str:
    return "turning_bands_sim({}) -> Turning bands simulation."
