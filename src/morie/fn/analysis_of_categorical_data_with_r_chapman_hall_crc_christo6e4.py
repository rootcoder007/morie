"""Joint probability of logistic responses (sufficiency form).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_4"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_4(b, x, y):
    """Joint probability of logistic responses (sufficiency form)

    Formula: P(Y1..Yn) = exp(sum y Xb)/prod(1 + exp(Xb)) (log form)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.4).
    """
    value = _acd.logistic_joint_probability(b, x, y)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.4)"
    return RichResult(
        title='Joint probability of logistic responses (sufficiency form)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e4: P(Y1..Yn) = exp(sum y Xb)/prod(1 + exp(Xb)) (log form) [Bilder & Loughin 2025, eq. 6.4]'
