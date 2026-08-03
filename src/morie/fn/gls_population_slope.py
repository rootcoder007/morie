"""Population GLS regression coefficient.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["gls_population_slope"]


def gls_population_slope(x, z, sigma2):
    """Population GLS regression coefficient

    Formula: b = (sum x x^T/sig2)^-1 sum x z/sig2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.4).
    """
    arr = np.asarray(_brus.gls_population_slope(x, z, sigma2), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.4)"
    return RichResult(
        title='Population GLS regression coefficient',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e4: b = (sum x x^T/sig2)^-1 sum x z/sig2 [Brus 2022, eq. 10.4]'
