"""Spatial panel random effects"""

from . import _array_core as np

from ._containers import SpatialResult


def spatial_panel_re(data, *, method="default"):
    """Spatial panel random effects

    Returns
    -------
    SpatialResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    mu = float(x.mean())
    stat = float(np.var(x)) if n > 1 else 0.0
    return SpatialResult(
        name="xrspr",
        statistic=stat,
        extra={"mean": mu, "n": n},
    )


spat = spatial_panel_re


def cheatsheet() -> str:
    return "spatial_panel_re({}) -> Spatial panel random effects"


# compact alias per ledger/NAMING.md
spatialpanelre = spatial_panel_re
