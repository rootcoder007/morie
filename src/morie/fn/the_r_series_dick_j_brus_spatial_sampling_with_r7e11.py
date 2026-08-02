"""Optimal number of PSUs for a budget.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_11"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_11(s_w, s_b, c1, c2, c_max):
    """Optimal number of PSUs for a budget

    Formula: n = C_max S_b/(S_w sqrt(c1 c2) + S_b c1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.11).
    """
    value = _brus.twostage_optimal_n_budget(s_w, s_b, c1, c2, c_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.11)"
    return RichResult(
        title='Optimal number of PSUs for a budget',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e11: n = C_max S_b/(S_w sqrt(c1 c2) + S_b c1) [Brus 2022, eq. 7.11]'
