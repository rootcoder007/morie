"""Expected squared prediction shift E[tau2].

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_5"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_5(cov_theta, dlam_dtheta, a):
    """Expected squared prediction shift E[tau2]

    Formula: E[tau2] = sum_ij Cov(th_i, th_j) dlam^T/dth_i A dlam/dth_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (24.5).
    """
    value = _brus.expected_tau2(cov_theta, dlam_dtheta, a)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (24.5)"
    return RichResult(
        title='Expected squared prediction shift E[tau2]',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r24e5: E[tau2] = sum_ij Cov(th_i, th_j) dlam^T/dth_i A dlam/dth_j [Brus 2022, eq. 24.5]'
