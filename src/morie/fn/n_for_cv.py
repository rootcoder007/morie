"""Required n for a relative error target.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["n_for_cv"]


def n_for_cv(u_crit, cv_star, r_max):
    """Required n for a relative error target

    Formula: n = (u cv*/r_max)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.10).
    """
    value = _brus.n_for_cv(u_crit, cv_star, r_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.10)"
    return RichResult(
        title='Required n for a relative error target',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e10: n = (u cv*/r_max)^2 [Brus 2022, eq. 12.10]'


# compact alias per ledger/NAMING.md
nforcv = n_for_cv
