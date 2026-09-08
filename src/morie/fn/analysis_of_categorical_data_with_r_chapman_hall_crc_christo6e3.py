"""MLE of the true prevalence.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_3(pi, se, sp):
    """MLE of the true prevalence

    Formula: pi_tilde_hat = (pi_hat + Sp - 1)/(Se + Sp - 1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.3).
    """
    value = _acd.prevalence_from_apparent(pi, se, sp)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.3)"
    return RichResult(
        title='MLE of the true prevalence',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e3: pi_tilde_hat = (pi_hat + Sp - 1)/(Se + Sp - 1) [Bilder & Loughin 2025, eq. 6.3]'
