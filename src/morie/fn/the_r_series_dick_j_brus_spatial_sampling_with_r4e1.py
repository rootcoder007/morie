"""Stratified estimator zbar_hat = sum w_h zbar_hat_h.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_1"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_1(stratum_means, stratum_weights):
    """Stratified estimator zbar_hat = sum w_h zbar_hat_h

    Formula: zbar_hat = sum_h w_h zbar_hat_h

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (4.1).
    """
    value = _brus.stratified_mean(stratum_means, stratum_weights)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (4.1)"
    return RichResult(
        title='Stratified estimator zbar_hat = sum w_h zbar_hat_h',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r4e1: zbar_hat = sum_h w_h zbar_hat_h [Brus 2022, eq. 4.1]'
