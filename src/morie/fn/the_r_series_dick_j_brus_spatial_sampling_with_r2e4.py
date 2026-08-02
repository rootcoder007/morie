"""Horvitz-Thompson estimator of the population mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_4(z, pi, n_population):
    """Horvitz-Thompson estimator of the population mean

    Formula: zbar_hat_pi = (1/N) sum_{k in S} z_k/pi_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (2.4).
    """
    value = _brus.ht_mean(z, pi, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (2.4)"
    return RichResult(
        title='Horvitz-Thompson estimator of the population mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r2e4: zbar_hat_pi = (1/N) sum_{k in S} z_k/pi_k [Brus 2022, eq. 2.4]'
