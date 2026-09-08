"""OLS trend weights over survey times.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["trend_weights"]


def trend_weights(times):
    """OLS trend weights over survey times

    Formula: w_j = (t_j - tbar)/sum(t - tbar)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (15.4).
    """
    arr = np.asarray(_brus.trend_weights(times), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (15.4)"
    return RichResult(
        title='OLS trend weights over survey times',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r15e4: w_j = (t_j - tbar)/sum(t - tbar)^2 [Brus 2022, eq. 15.4]'


# compact alias per ledger/NAMING.md
trendweights = trend_weights
