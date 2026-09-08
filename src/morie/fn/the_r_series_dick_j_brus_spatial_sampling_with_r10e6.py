"""Sample-weighted GLS regression coefficient.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_6"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_6(x, z, sigma2, pi):
    """Sample-weighted GLS regression coefficient

    Formula: b_hat = (sum x x^T/(sig2 pi))^-1 sum x z/(sig2 pi)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.6).
    """
    arr = np.asarray(_brus.gls_sample_slope(x, z, sigma2, pi), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.6)"
    return RichResult(
        title='Sample-weighted GLS regression coefficient',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e6: b_hat = (sum x x^T/(sig2 pi))^-1 sum x z/(sig2 pi) [Brus 2022, eq. 10.6]'
