"""OLS prediction variance at a new point.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_20_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_20_equation_3(sigma2_eps, x0, x):
    """OLS prediction variance at a new point

    Formula: V_hat(Z(s0)) = sig2_eps (1 + x0^T (X^T X)^-1 x0)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (20.3).
    """
    value = _brus.ols_prediction_variance(sigma2_eps, x0, x)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (20.3)"
    return RichResult(
        title='OLS prediction variance at a new point',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r20e3: V_hat(Z(s0)) = sig2_eps (1 + x0^T (X^T X)^-1 x0) [Brus 2022, eq. 20.3]'
