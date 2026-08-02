"""Logit form log(pi/(1-pi)) = b0 + b1 x1 + ... + bp xp.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_3(p):
    """Logit form log(pi/(1-pi)) = b0 + b1 x1 + ... + bp xp

    Formula: log(pi/(1-pi)) = Xb

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.3).
    """
    value = _acd.logit_form(p)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.3)"
    return RichResult(
        title='Logit form log(pi/(1-pi)) = b0 + b1 x1 + ... + bp xp',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e3: log(pi/(1-pi)) = Xb [Bilder & Loughin 2025, eq. 2.3]'
