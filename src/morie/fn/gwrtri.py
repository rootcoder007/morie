# morie.fn -- function file (rootcoder007/morie)
"""GWR tri-cube kernel weights."""

from __future__ import annotations

from ._containers import SpatialResult
from .gwrbas import gwr_kernel_weights


def gwrtri(dists, bw=0.5, adaptive: bool = False) -> SpatialResult:
    """GWR tri-cube kernel weights of distances (= ``GWmodel::gw.weight`` kernel ``tricube``).

    ``local_values`` are the weights; ``statistic`` is their sum.
    """
    w = gwr_kernel_weights(dists, bw, "tricube", adaptive)
    return SpatialResult(name="gwrtri", statistic=sum(w), p_value=None, local_values=w)


gwrtri_fn = gwrtri


def cheatsheet() -> str:
    return "gwrtri(dists, bw) -> GWR tri-cube kernel weights."
