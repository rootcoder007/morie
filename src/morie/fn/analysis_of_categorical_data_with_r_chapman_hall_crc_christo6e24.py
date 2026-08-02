"""Bayes estimate as a weighted average of MLE and prior mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_24"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_24(w, n, a, b):
    """Bayes estimate as a weighted average of MLE and prior mean

    Formula: pi_B = (n/(n+a+b)) pi_hat + ((a+b)/(n+a+b)) E(pi)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.24).
    """
    value = _acd.bayes_estimate_binomial(w, n, a, b)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.24)"
    return RichResult(
        title='Bayes estimate as a weighted average of MLE and prior mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e24: pi_B = (n/(n+a+b)) pi_hat + ((a+b)/(n+a+b)) E(pi) [Bilder & Loughin 2025, eq. 6.24]'
