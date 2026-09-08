"""Second-phase residual variance S2_hat(e).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["s2_residuals"]


def s2_residuals(e, n):
    """Second-phase residual variance S2_hat(e)

    Formula: S2_hat(e) = (1/(n2-1)) sum e_k^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (11.8).
    """
    value = _brus.s2_residuals(e, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (11.8)"
    return RichResult(
        title='Second-phase residual variance S2_hat(e)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r11e8: S2_hat(e) = (1/(n2-1)) sum e_k^2 [Brus 2022, eq. 11.8]'


# compact alias per ledger/NAMING.md
s2residuals = s2_residuals
