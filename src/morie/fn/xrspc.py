"""Spatial Poisson count model"""

from . import _array_core as np

from ._containers import SpatialResult


def spatial_poisson(data, *, method="default"):
    """Spatial Poisson count model

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrspc",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_poisson


def cheatsheet() -> str:
    return "spatial_poisson({}) -> Spatial Poisson count model"


# compact alias per ledger/NAMING.md
spatialpoisson = spatial_poisson
