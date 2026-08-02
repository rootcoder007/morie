"""Regression estimator of the total.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_2(t_pi_z, t_x_true, t_pi_x, b_hat):
    """Regression estimator of the total

    Formula: t_regr = t_pi(z) + b_hat (t(x) - t_pi(x))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (9.2).
    """
    value = _brus.regression_total(t_pi_z, t_x_true, t_pi_x, b_hat)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (9.2)"
    return RichResult(
        title='Regression estimator of the total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r9e2: t_regr = t_pi(z) + b_hat (t(x) - t_pi(x)) [Brus 2022, eq. 9.2]'
