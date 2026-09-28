# morie.fn -- function file (rootcoder007/morie)
"""GWR Gaussian kernel weights."""

from __future__ import annotations

from ._containers import SpatialResult
from .gwrbas import gwr_kernel_weights


def gwrgaus(dists, bw=0.5, adaptive: bool = False) -> SpatialResult:
    """GWR Gaussian kernel weights of distances (= ``GWmodel::gw.weight`` kernel ``gaussian``).

    ``local_values`` are the weights; ``statistic`` is their sum.
    """
    w = gwr_kernel_weights(dists, bw, "gaussian", adaptive)
    return SpatialResult(name="gwrgaus", statistic=sum(w), p_value=None, local_values=w)


gwrgaus_fn = gwrgaus


def cheatsheet() -> str:
    return "gwrgaus(dists, bw) -> GWR Gaussian kernel weights."
