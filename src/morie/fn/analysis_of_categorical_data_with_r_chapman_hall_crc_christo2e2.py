"""Logistic regression model pi = exp(Xb)/(1 + exp(Xb)).

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_2"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_2(b0, bs, xs):
    """Logistic regression model pi = exp(Xb)/(1 + exp(Xb))

    Formula: pi = exp(b0 + b1 x1 + ... + bp xp)/(1 + exp(...))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.2).
    """
    value = _acd.logistic_pi(b0, bs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.2)"
    return RichResult(
        title='Logistic regression model pi = exp(Xb)/(1 + exp(Xb))',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e2: pi = exp(b0 + b1 x1 + ... + bp xp)/(1 + exp(...)) [Bilder & Loughin 2025, eq. 2.2]'
