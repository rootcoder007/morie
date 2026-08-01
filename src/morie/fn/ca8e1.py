"""Generic noncentrality delta = mean(pop stat) - mean(null stat).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_8_equation_1"]


def ca_chapter_8_equation_1(mean_population, mean_null):
    """Generic noncentrality delta = mean(pop stat) - mean(null stat)

    Formula: delta = mean test statistic (population) - mean (null)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.1
    """
    value = _ca_crim.noncentrality_delta_generic(mean_population, mean_null)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.1)"
    return RichResult(
        title='Generic noncentrality delta = mean(pop stat) - mean(null stat)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e1: delta = mean test statistic (population) - mean (null) [Weisburd et al. 2022, eq. 8.1]'
