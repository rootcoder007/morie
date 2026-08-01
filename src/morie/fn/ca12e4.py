"""Spatial lag (SAR) model y = rho W y + x beta + e.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_12_equation_4"]


def ca_chapter_12_equation_4(rho, w, xb, e):
    """Spatial lag (SAR) model y = rho W y + x beta + e

    Formula: y = rho W y + x beta + e  (solved via y = (I - rho W)^-1 (xb + e))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.12 eq.12.4
    """
    payload = dict({"y": _ca_crim.spatial_lag_reduced_form(rho, w, xb, e).tolist(), "value": float(_ca_crim.spatial_lag_reduced_form(rho, w, xb, e)[0])})
    value = payload['value']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (12.4)"
    return RichResult(
        title='Spatial lag (SAR) model y = rho W y + x beta + e',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca12e4: y = rho W y + x beta + e  (solved via y = (I - rho W)^-1 (xb + e)) [Weisburd et al. 2022, eq. 12.4]'
