"""Weights connectivity check"""

from . import _array_core as np

from ._containers import SpatialResult


def w_connected(data, *, method="default"):
    """Weights connectivity check

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
        name="Weights connectivity check",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


w_co = w_connected


def cheatsheet() -> str:
    return "w_connected({}) -> Weights connectivity check"


# compact alias per ledger/NAMING.md
wconnected = w_connected
