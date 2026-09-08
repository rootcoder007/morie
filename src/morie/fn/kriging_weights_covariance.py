"""Ordinary kriging equation system.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["kriging_weights_covariance"]


def kriging_weights_covariance(cov_ss, cov_s0):
    """Ordinary kriging equation system

    Formula: sum lam_j C(s_i, s_j) + nu = C(s_i, s0); sum lam_j = 1

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'nu' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.4).
    """
    payload = dict((lambda s: {"lam": s["lam"].tolist(), "nu": s["nu"], "value": s["nu"]})(_brus.kriging_weights_covariance(cov_ss, cov_s0)))
    value = payload['nu']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.4)"
    return RichResult(
        title='Ordinary kriging equation system',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e4: sum lam_j C(s_i, s_j) + nu = C(s_i, s0); sum lam_j = 1 [Brus 2022, eq. 21.4]'
