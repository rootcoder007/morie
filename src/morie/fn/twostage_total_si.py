"""Two-stage total with SI of PSUs.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["twostage_total_si"]


def twostage_total_si(psu_total_estimates, n_psu_population):
    """Two-stage total with SI of PSUs

    Formula: t_hat = (N/n) sum t_hat_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.13).
    """
    value = _brus.twostage_total_si(psu_total_estimates, n_psu_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.13)"
    return RichResult(
        title='Two-stage total with SI of PSUs',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e13: t_hat = (N/n) sum t_hat_j [Brus 2022, eq. 7.13]'
