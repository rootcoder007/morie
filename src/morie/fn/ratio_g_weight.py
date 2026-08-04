"""Constant g-weight of the ratio estimator.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["ratio_g_weight"]


def ratio_g_weight(t_x_true, t_pi_x):
    """Constant g-weight of the ratio estimator

    Formula: g = t(x)/t_pi(x)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.27).
    """
    value = _brus.ratio_g_weight(t_x_true, t_pi_x)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.27)"
    return RichResult(
        title='Constant g-weight of the ratio estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e27: g = t(x)/t_pi(x) [Brus 2022, eq. 10.27]'


# compact alias per ledger/NAMING.md
ratiogweight = ratio_g_weight
