"""Two-phase sampling for stratification: variance.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_5"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_equation_5(n1h, n1, s2_2h, n2h, zbar_2h, zbar_hat):
    """Two-phase sampling for stratification: variance

    Formula: V_hat = sum (n1h/n1)^2 S2_2h/n2h + (1/n1) sum (n1h/n1)(zbar_2h - zbar_hat)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (11.5).
    """
    value = _brus.twophase_stratified_variance(n1h, n1, s2_2h, n2h, zbar_2h, zbar_hat)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (11.5)"
    return RichResult(
        title='Two-phase sampling for stratification: variance',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r11e5: V_hat = sum (n1h/n1)^2 S2_2h/n2h + (1/n1) sum (n1h/n1)(zbar_2h - zbar_hat)^2 [Brus 2022, eq. 11.5]'
