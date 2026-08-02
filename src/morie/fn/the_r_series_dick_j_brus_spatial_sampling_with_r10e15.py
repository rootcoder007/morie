"""OLS coefficient from the sample (equal weights).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_15"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_15(x, z):
    """OLS coefficient from the sample (equal weights)

    Formula: b_hat = (sum x x^T)^-1 sum x z

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.15).
    """
    arr = np.asarray(_brus.gls_sample_slope(x, z, np.ones(len(np.atleast_1d(np.asarray(z)))), np.ones(len(np.atleast_1d(np.asarray(z))))), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.15)"
    return RichResult(
        title='OLS coefficient from the sample (equal weights)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e15: b_hat = (sum x x^T)^-1 sum x z [Brus 2022, eq. 10.15]'
