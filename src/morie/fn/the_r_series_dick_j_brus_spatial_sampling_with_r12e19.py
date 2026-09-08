"""Bayesian average coverage criterion (continuous data).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_19"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_19(coverages, probs, alpha):
    """Bayesian average coverage criterion (continuous data)

    Formula: E[coverage of (v, v + l_max)] >= 1 - alpha

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'expected_coverage' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.19).
    """
    payload = dict(_brus.average_coverage_criterion(coverages, probs, alpha))
    value = payload['expected_coverage']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.19)"
    return RichResult(
        title='Bayesian average coverage criterion (continuous data)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e19: E[coverage of (v, v + l_max)] >= 1 - alpha [Brus 2022, eq. 12.19]'
