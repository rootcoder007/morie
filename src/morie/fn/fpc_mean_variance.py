"""Design-based variance with finite population correction.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["fpc_mean_variance"]


def fpc_mean_variance(s2, n, n_population):
    """Design-based variance with finite population correction

    Formula: V(zbar_hat) = (1 - n/N) S2/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (26.5).
    """
    value = _brus.fpc_mean_variance(s2, n, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (26.5)"
    return RichResult(
        title='Design-based variance with finite population correction',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r26e5: V(zbar_hat) = (1 - n/N) S2/n [Brus 2022, eq. 26.5]'
