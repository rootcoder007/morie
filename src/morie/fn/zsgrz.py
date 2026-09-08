"""Zonal grid statistics"""

from . import _array_core as np

from ._containers import SpatialResult


def grid_zonal(data, *, method="default"):
    """Zonal grid statistics

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="Zonal grid statistics",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


grid = grid_zonal


def cheatsheet() -> str:
    return "grid_zonal({}) -> Zonal grid statistics"


# compact alias per ledger/NAMING.md
gridzonal = grid_zonal
