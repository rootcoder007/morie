"""Power location t_beta = delta - t_CV (with power estimate).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["power_from_delta_t"]


def power_from_delta_t(delta, t_cv, df):
    """Power location t_beta = delta - t_CV (with power estimate)

    Formula: t_beta = delta - t_CV; power = P(T_nc(delta) > t_CV)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 't_beta' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.3
    """
    payload = dict(_ca_crim.power_from_delta_t(delta, t_cv, df))
    value = payload['t_beta']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.3)"
    return RichResult(
        title='Power location t_beta = delta - t_CV (with power estimate)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e3: t_beta = delta - t_CV; power = P(T_nc(delta) > t_CV) [Weisburd et al. 2022, eq. 8.3]'
