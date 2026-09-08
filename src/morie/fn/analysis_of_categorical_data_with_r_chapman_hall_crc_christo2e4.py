"""Logistic log-likelihood (product form of eq 2.1).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_4"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_4(b, x, y):
    """Logistic log-likelihood (product form of eq 2.1)

    Formula: log L = sum y_i log(pi_i) + (1-y_i) log(1-pi_i)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.4).
    """
    value = _acd.logistic_loglik(b, x, y)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.4)"
    return RichResult(
        title='Logistic log-likelihood (product form of eq 2.1)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e4: log L = sum y_i log(pi_i) + (1-y_i) log(1-pi_i) [Bilder & Loughin 2025, eq. 2.4]'
