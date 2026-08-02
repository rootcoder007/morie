"""Model-expected STSI variance from mean semivariances.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_5"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_5(gamma_bar_h, weights, n_h):
    """Model-expected STSI variance from mean semivariances

    Formula: E_xi{V_STSI} = sum w_h^2 gammabar_h/n_h

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.5).
    """
    value = _brus.mean_semivariance_stsi_variance(gamma_bar_h, weights, n_h)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.5)"
    return RichResult(
        title='Model-expected STSI variance from mean semivariances',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e5: E_xi{V_STSI} = sum w_h^2 gammabar_h/n_h [Brus 2022, eq. 13.5]'
