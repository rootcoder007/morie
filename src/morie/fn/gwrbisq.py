# morie.fn -- function file (rootcoder007/morie)
"""GWR bisquare kernel weights."""

from __future__ import annotations

from ._containers import SpatialResult
from .gwrbas import gwr_kernel_weights


def gwrbisq(dists, bw=0.5, adaptive: bool = False) -> SpatialResult:
    """GWR bisquare kernel weights of distances (= ``GWmodel::gw.weight`` kernel ``bisquare``).

    ``local_values`` are the weights; ``statistic`` is their sum.
    """
    w = gwr_kernel_weights(dists, bw, "bisquare", adaptive)
    return SpatialResult(name="gwrbisq", statistic=sum(w), p_value=None, local_values=w)


gwrbisq_fn = gwrbisq


def cheatsheet() -> str:
    return "gwrbisq(dists, bw) -> GWR bisquare kernel weights."
