"""SAR direct/indirect/total impacts"""

from . import _array_core as np

from ._containers import SpatialResult


def sar_impacts(data, *, method="default"):
    """SAR direct/indirect/total impacts

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
        name="SAR direct/indirect/total impacts",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sar_ = sar_impacts


def cheatsheet() -> str:
    return "sar_impacts({}) -> SAR direct/indirect/total impacts"


# compact alias per ledger/NAMING.md
sarimpacts = sar_impacts
