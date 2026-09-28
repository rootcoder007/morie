# morie.fn -- function file (rootcoder007/morie)
"""GWR basic model fit (Brunsdon, Fotheringham and Charlton 1996)."""

from __future__ import annotations

from ._containers import SpatialResult
from .gwrbas import gwr_basic


def gwrfit(y, X, coords, bw=1.0, kernel: str = "bisquare", adaptive: bool = False) -> SpatialResult:
    """Fit a basic GWR (:func:`~morie.fn.gwrbas.gwr_basic`, = ``GWmodel::gwr.basic``).

    ``statistic`` is the AICc; ``extra`` carries the full fit.
    """
    r = gwr_basic(y, X, coords, bw, kernel=kernel, adaptive=adaptive)
    return SpatialResult(name="gwrfit", statistic=r["diagnostics"]["AICc"], p_value=None, extra=dict(r))


gwrfit_fn = gwrfit


def cheatsheet() -> str:
    return "gwrfit(y, X, coords, bw) -> basic GWR fit (GWmodel::gwr.basic)."
