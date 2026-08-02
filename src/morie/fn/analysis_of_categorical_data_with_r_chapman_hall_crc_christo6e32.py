"""Group-testing logistic regression.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_32"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_32(b0, bs, xs):
    """Group-testing logistic regression

    Formula: logit(pi_tilde_i(k)) = b0 + b1 x1 + ... + bp xp

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.32).
    """
    value = _acd.group_testing_logit(b0, bs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.32)"
    return RichResult(
        title='Group-testing logistic regression',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e32: logit(pi_tilde_i(k)) = b0 + b1 x1 + ... + bp xp [Bilder & Loughin 2025, eq. 6.32]'
