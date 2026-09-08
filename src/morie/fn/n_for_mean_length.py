"""Required n for a confidence interval length on a mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["n_for_mean_length"]


def n_for_mean_length(u_crit, s_star, l_max):
    """Required n for a confidence interval length on a mean

    Formula: n = (u S*/(l_max/2))^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.7).
    """
    value = _brus.n_for_mean_length(u_crit, s_star, l_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.7)"
    return RichResult(
        title='Required n for a confidence interval length on a mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e7: n = (u S*/(l_max/2))^2 [Brus 2022, eq. 12.7]'


# compact alias per ledger/NAMING.md
nformeanlength = n_for_mean_length
