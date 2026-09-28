# morie.fn -- function file (rootcoder007/morie)
"""GWR t-values for local coefficients."""

from __future__ import annotations

from ._containers import SpatialResult
from .gwrbas import gwr_basic


def gwrtvl(y, X, coords, bw=0.5, kernel: str = "bisquare", adaptive: bool = False) -> SpatialResult:
    """Local t-values ``beta_i / se_i`` of a GWR (= ``GWmodel::gwr.basic`` ``*_TV``).

    ``statistic`` is the largest absolute t-value; ``extra["tvalues"]`` is n x p.
    """
    r = gwr_basic(y, X, coords, bw, kernel=kernel, adaptive=adaptive)
    tv = r["tvalues"]
    finite = [abs(v) for row in tv for v in row if v == v]
    return SpatialResult(
        name="gwrtvl",
        statistic=max(finite) if finite else float("nan"),
        p_value=None,
        extra={"tvalues": tv, "betas": r["betas"], "se": r["se"]},
    )


gwrtvl_fn = gwrtvl


def cheatsheet() -> str:
    return "gwrtvl(y, X, coords, bw) -> GWR local t-values."
