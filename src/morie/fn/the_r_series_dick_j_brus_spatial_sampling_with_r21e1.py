"""Stationary Gaussian process model for kriging.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_1"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_1(mu, cov):
    """Stationary Gaussian process model for kriging

    Formula: Z(s) = mu(s) + eps(s), eps ~ N(0, sigma2), Cov = C(h)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'n' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.1).
    """
    payload = dict({"n": _brus.gaussian_process_model(mu, cov)["n"], "value": float(_brus.gaussian_process_model(mu, cov)["n"])})
    value = payload['n']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.1)"
    return RichResult(
        title='Stationary Gaussian process model for kriging',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e1: Z(s) = mu(s) + eps(s), eps ~ N(0, sigma2), Cov = C(h) [Brus 2022, eq. 21.1]'
