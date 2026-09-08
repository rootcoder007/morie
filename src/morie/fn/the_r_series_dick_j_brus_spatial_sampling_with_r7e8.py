"""Between-PSU variance S2_hat(zbar).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_8(primary_unit_means):
    """Between-PSU variance S2_hat(zbar)

    Formula: S2_hat = sum(zbar_j - zbarbar)^2/(n-1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 's2_psu' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.8).
    """
    payload = dict(_brus.twostage_variance_estimator(primary_unit_means))
    value = payload['s2_psu']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.8)"
    return RichResult(
        title='Between-PSU variance S2_hat(zbar)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e8: S2_hat = sum(zbar_j - zbarbar)^2/(n-1) [Brus 2022, eq. 7.8]'
