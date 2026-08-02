"""Stratum sample mean zbar_hat_h = (1/n_h) sum z_k.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_2(z_h):
    """Stratum sample mean zbar_hat_h = (1/n_h) sum z_k

    Formula: zbar_hat_h = (1/n_h) sum_{k in S_h} z_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (4.2).
    """
    value = float(np.asarray(z_h, dtype=float).mean())
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (4.2)"
    return RichResult(
        title='Stratum sample mean zbar_hat_h = (1/n_h) sum z_k',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r4e2: zbar_hat_h = (1/n_h) sum_{k in S_h} z_k [Brus 2022, eq. 4.2]'
