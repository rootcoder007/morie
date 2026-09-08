"""Heteroscedastic through-the-origin working model.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_24(beta, x_k, sigma2):
    """Heteroscedastic through-the-origin working model

    Formula: Z(x_k) = beta x_k + eps_k with sigma2(eps_k) = sigma2 x_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'prediction' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.24).
    """
    payload = dict({"prediction": float(beta) * float(x_k), "variance": float(sigma2) * float(x_k), "value": float(beta) * float(x_k)})
    value = payload['prediction']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.24)"
    return RichResult(
        title='Heteroscedastic through-the-origin working model',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e24: Z(x_k) = beta x_k + eps_k with sigma2(eps_k) = sigma2 x_k [Brus 2022, eq. 10.24]'
