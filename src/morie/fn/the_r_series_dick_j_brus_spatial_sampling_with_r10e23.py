"""Ratio estimator of the total.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_23"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_23(t_pi_z, t_pi_x, t_x_true):
    """Ratio estimator of the total

    Formula: t_ratio = (t_pi(z)/t_pi(x)) t(x)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.23).
    """
    value = _brus.ratio_total(t_pi_z, t_pi_x, t_x_true)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.23)"
    return RichResult(
        title='Ratio estimator of the total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e23: t_ratio = (t_pi(z)/t_pi(x)) t(x) [Brus 2022, eq. 10.23]'
