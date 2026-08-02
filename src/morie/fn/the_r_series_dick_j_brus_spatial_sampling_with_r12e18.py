"""Posterior interval with coverage 1 - alpha.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_18(v, l, z, n, c, d):
    """Posterior interval with coverage 1 - alpha

    Formula: integral over (v, v + l) of the posterior = 1 - alpha

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.18).
    """
    value = _brus.beta_posterior_interval_prob(v, l, z, n, c, d)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.18)"
    return RichResult(
        title='Posterior interval with coverage 1 - alpha',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e18: integral over (v, v + l) of the posterior = 1 - alpha [Brus 2022, eq. 12.18]'
