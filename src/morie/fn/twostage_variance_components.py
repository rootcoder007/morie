"""True two-stage variance S2_b/n + S2_w/(n m).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["twostage_variance_components"]


def twostage_variance_components(s2_between, s2_within, n, m):
    """True two-stage variance S2_b/n + S2_w/(n m)

    Formula: V = S2_b/n + S2_w/(n m)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.3).
    """
    value = _brus.twostage_variance_components(s2_between, s2_within, n, m)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.3)"
    return RichResult(
        title='True two-stage variance S2_b/n + S2_w/(n m)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e3: V = S2_b/n + S2_w/(n m) [Brus 2022, eq. 7.3]'
