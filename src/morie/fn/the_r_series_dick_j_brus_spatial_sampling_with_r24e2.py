"""REML Fisher information of covariance parameters.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_2(a, da_list):
    """REML Fisher information of covariance parameters

    Formula: I_ij = 0.5 Tr(A^-1 dA_i A^-1 dA_j)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (24.2).
    """
    arr = np.asarray(_brus.fisher_information_reml(a, da_list), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (24.2)"
    return RichResult(
        title='REML Fisher information of covariance parameters',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r24e2: I_ij = 0.5 Tr(A^-1 dA_i A^-1 dA_j) [Brus 2022, eq. 24.2]'
