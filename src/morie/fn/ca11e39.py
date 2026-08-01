"""Upper confidence bound ybar + z_CV se.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_39"]


def ca_chapter_11_equation_39(ys, ws, z_cv):
    """Upper confidence bound ybar + z_CV se

    Formula: ybar_upper = ybar + z_CV se_ybar

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'upper' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.39
    """
    payload = dict((lambda m: _ca_crim.coef_ci(m['mean'], m['se'], z_cv))(_ca_crim.mean_effect_size(ys, ws)))
    value = payload['upper']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.39)"
    return RichResult(
        title='Upper confidence bound ybar + z_CV se',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e39: ybar_upper = ybar + z_CV se_ybar [Weisburd et al. 2022, eq. 11.39]'
