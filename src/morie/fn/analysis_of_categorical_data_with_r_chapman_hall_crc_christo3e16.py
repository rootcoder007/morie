"""Non-proportional odds model.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_16"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_16(bj0, bjs, xs):
    """Non-proportional odds model

    Formula: logit(P(Y <= j)) = bj0 + bj1 x1 + ... + bjp xp

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.16).
    """
    value = _acd.nonproportional_odds_logit(bj0, bjs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.16)"
    return RichResult(
        title='Non-proportional odds model',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e16: logit(P(Y <= j)) = bj0 + bj1 x1 + ... + bjp xp [Bilder & Loughin 2025, eq. 3.16]'
