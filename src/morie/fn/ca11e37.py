"""z-test for the mean effect size z = ybar / se.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_37"]


def ca_chapter_11_equation_37(ys, ws):
    """z-test for the mean effect size z = ybar / se

    Formula: z = ybar / se_ybar

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'z' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.37
    """
    payload = dict(_ca_crim.mean_effect_size(ys, ws))
    value = payload['z']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.37)"
    return RichResult(
        title='z-test for the mean effect size z = ybar / se',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e37: z = ybar / se_ybar [Weisburd et al. 2022, eq. 11.37]'
