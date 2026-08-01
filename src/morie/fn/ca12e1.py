"""Moran's I spatial autocorrelation index.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_12_equation_1"]


def ca_chapter_12_equation_1(x, w):
    """Moran's I spatial autocorrelation index

    Formula: I = n sum_ij w_ij (x_i - xbar)(x_j - xbar) / (W sum(x - xbar)^2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.12 eq.12.1
    """
    value = _ca_crim.morans_i(x, w)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (12.1)"
    return RichResult(
        title="Moran's I spatial autocorrelation index",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca12e1: I = n sum_ij w_ij (x_i - xbar)(x_j - xbar) / (W sum(x - xbar)^2) [Weisburd et al. 2022, eq. 12.1]'
