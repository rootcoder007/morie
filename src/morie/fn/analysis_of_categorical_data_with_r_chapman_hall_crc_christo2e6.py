"""Likelihood ratio Lambda = max L under H0 / max L under H0 or Ha.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_6"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_6(loglik_null, loglik_full):
    """Likelihood ratio Lambda = max L under H0 / max L under H0 or Ha

    Formula: -2 log(Lambda) = -2 (LL0 - LLa)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.6).
    """
    value = _acd.lrt_statistic(loglik_null, loglik_full)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.6)"
    return RichResult(
        title='Likelihood ratio Lambda = max L under H0 / max L under H0 or Ha',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e6: -2 log(Lambda) = -2 (LL0 - LLa) [Bilder & Loughin 2025, eq. 2.6]'
