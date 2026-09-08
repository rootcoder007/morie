"""Two-phase sampling for regression: variance.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["twophase_regression_variance"]


def twophase_regression_variance(s2_z, n1, s2_e, n2, n_population):
    """Two-phase sampling for regression: variance

    Formula: V_hat = (1-n1/N) S2(z)/n1 + (1-n2/n1) S2(e)/n2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (11.7).
    """
    value = _brus.twophase_regression_variance(s2_z, n1, s2_e, n2, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (11.7)"
    return RichResult(
        title='Two-phase sampling for regression: variance',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r11e7: V_hat = (1-n1/N) S2(z)/n1 + (1-n2/n1) S2(e)/n2 [Brus 2022, eq. 11.7]'
