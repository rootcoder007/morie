# morie.fn -- function file (rootcoder007/morie)
"""GWR bandwidth selection by AICc."""

from __future__ import annotations

from ._containers import SpatialResult
from .spgwrb import schabenberger_gwr_bandwidth


def gwrbw(y, X, coords, kernel: str = "gaussian", adaptive: bool = False) -> SpatialResult:
    """Bandwidth minimising the GWR AICc (:func:`~morie.fn.spgwrb.schabenberger_gwr_bandwidth`).

    ``statistic`` is the chosen bandwidth.
    """
    r = schabenberger_gwr_bandwidth(X, y, coords, kernel=kernel, criterion="aicc", adaptive=adaptive)
    bw = r["optimal_bandwidth"]
    return SpatialResult(name="gwrbw", statistic=float(bw), p_value=None, extra=dict(r))


gwrbw_fn = gwrbw


def cheatsheet() -> str:
    return "gwrbw(y, X, coords) -> GWR bandwidth by AICc."
