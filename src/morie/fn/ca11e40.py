"""Homogeneity Q = sum w (y - ybar)^2.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_40"]


def ca_chapter_11_equation_40(ys, ws):
    """Homogeneity Q = sum w (y - ybar)^2

    Formula: Q = sum w_i (y_i - ybar)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'q' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.40
    """
    payload = dict(_ca_crim.q_statistic(ys, ws))
    value = payload['q']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.40)"
    return RichResult(
        title='Homogeneity Q = sum w (y - ybar)^2',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e40: Q = sum w_i (y_i - ybar)^2 [Weisburd et al. 2022, eq. 11.40]'
