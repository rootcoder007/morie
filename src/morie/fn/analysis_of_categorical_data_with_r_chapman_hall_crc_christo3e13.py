"""polr() parameterization logit(P(Y <= j)) = bj0 - eta^T x.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_13"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_13(bj0, etas, xs):
    """polr() parameterization logit(P(Y <= j)) = bj0 - eta^T x

    Formula: logit(P(Y <= j)) = bj0 - eta1 x1 - ... - etap xp

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.13).
    """
    value = _acd.polr_parameterization(bj0, etas, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.13)"
    return RichResult(
        title='polr() parameterization logit(P(Y <= j)) = bj0 - eta^T x',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e13: logit(P(Y <= j)) = bj0 - eta1 x1 - ... - etap xp [Bilder & Loughin 2025, eq. 3.13]'
