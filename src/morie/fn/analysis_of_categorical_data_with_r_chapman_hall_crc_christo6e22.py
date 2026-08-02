"""Bayes rule P(B|A) = P(A|B)P(B)/P(A).

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_22"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_22(p_a_given_b, p_b, p_a_given_notb):
    """Bayes rule P(B|A) = P(A|B)P(B)/P(A)

    Formula: P(B|A) = P(A|B)P(B)/(P(A|B)P(B) + P(A|~B)P(~B))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.22).
    """
    value = _acd.bayes_rule(p_a_given_b, p_b, p_a_given_notb)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.22)"
    return RichResult(
        title='Bayes rule P(B|A) = P(A|B)P(B)/P(A)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e22: P(B|A) = P(A|B)P(B)/(P(A|B)P(B) + P(A|~B)P(~B)) [Bilder & Loughin 2025, eq. 6.22]'
