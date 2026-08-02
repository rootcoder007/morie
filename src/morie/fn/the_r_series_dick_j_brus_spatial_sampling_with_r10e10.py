"""Simple regression estimator under SI.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_10(zbar_sample, b_hat, xbar_true, xbar_sample):
    """Simple regression estimator under SI

    Formula: zbar_regr = zbar_S + b_hat (xbar - xbar_S)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.10).
    """
    value = _brus.regression_estimator_slopes(zbar_sample, [b_hat], [xbar_true], [xbar_sample])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.10)"
    return RichResult(
        title='Simple regression estimator under SI',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e10: zbar_regr = zbar_S + b_hat (xbar - xbar_S) [Brus 2022, eq. 10.10]'
