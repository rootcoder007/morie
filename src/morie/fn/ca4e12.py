"""Percent of correct predictions.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_4_equation_12"]


def ca_chapter_4_equation_12(n_correct, n_total):
    """Percent of correct predictions

    Formula: 100 n_correct / n_total

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.12
    """
    value = _ca_crim.percent_correct_predictions(n_correct, n_total)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.12)"
    return RichResult(
        title='Percent of correct predictions',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e12: 100 n_correct / n_total [Weisburd et al. 2022, eq. 4.12]'
