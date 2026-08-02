"""Beta probability density function.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_5"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_5(v, a, b):
    """Beta probability density function

    Formula: f(v; a, b) = Gamma(a+b)/(Gamma(a)Gamma(b)) v^(a-1)(1-v)^(b-1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.5).
    """
    value = _acd.beta_pdf(v, a, b)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.5)"
    return RichResult(
        title='Beta probability density function',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e5: f(v; a, b) = Gamma(a+b)/(Gamma(a)Gamma(b)) v^(a-1)(1-v)^(b-1) [Bilder & Loughin 2025, eq. 1.5]'
