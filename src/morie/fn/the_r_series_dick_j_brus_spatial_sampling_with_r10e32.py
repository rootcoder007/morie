"""Poststratification (ANOVA) working model estimator.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_32"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_32(group_means_sample, group_weights):
    """Poststratification (ANOVA) working model estimator

    Formula: Z_k = mu_g + eps_k -> zbar_pst = sum w_g zbar_S,g

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.32).
    """
    value = _brus.poststratified_mean(group_means_sample, group_weights)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.32)"
    return RichResult(
        title='Poststratification (ANOVA) working model estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e32: Z_k = mu_g + eps_k -> zbar_pst = sum w_g zbar_S,g [Brus 2022, eq. 10.32]'
