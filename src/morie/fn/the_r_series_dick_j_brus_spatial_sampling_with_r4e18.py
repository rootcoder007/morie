"""Linear cost model C = c0 + sum n_h c_h.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_18(c0, stratum_costs, stratum_sizes):
    """Linear cost model C = c0 + sum n_h c_h

    Formula: C = c0 + sum_h n_h c_h

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (4.18).
    """
    value = _brus.stratified_cost(c0, stratum_costs, stratum_sizes)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (4.18)"
    return RichResult(
        title='Linear cost model C = c0 + sum n_h c_h',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r4e18: C = c0 + sum_h n_h c_h [Brus 2022, eq. 4.18]'
