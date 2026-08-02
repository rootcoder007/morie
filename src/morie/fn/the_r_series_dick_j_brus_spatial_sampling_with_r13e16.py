"""Model-expected stratum variance (print eq 13.16).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_16"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_16(d2_upper_sum, n_h):
    """Model-expected stratum variance (print eq 13.16)

    Formula: E_xi[S2_h(z)] = (1/N_h^2) sum_{i<j} E_xi[d2_ij]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.16).
    """
    value = _brus.expected_stratum_variance(d2_upper_sum, n_h)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.16)"
    return RichResult(
        title='Model-expected stratum variance (print eq 13.16)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e16: E_xi[S2_h(z)] = (1/N_h^2) sum_{i<j} E_xi[d2_ij] [Brus 2022, eq. 13.16]'
