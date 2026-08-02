"""Bayesian average length criterion (continuous data).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_17"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_17(lengths, probs, l_max):
    """Bayesian average length criterion (continuous data)

    Formula: E[l(z, n)] <= l_max over the predictive distribution

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'expected_length' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.17).
    """
    payload = dict(_brus.average_length_criterion(lengths, probs, l_max))
    value = payload['expected_length']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.17)"
    return RichResult(
        title='Bayesian average length criterion (continuous data)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e17: E[l(z, n)] <= l_max over the predictive distribution [Brus 2022, eq. 12.17]'
