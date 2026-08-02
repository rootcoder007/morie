"""Prevalence from apparent probability, Se and Sp.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_1"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_1(pi, se, sp):
    """Prevalence from apparent probability, Se and Sp

    Formula: pi_tilde = (pi + Sp - 1)/(Se + Sp - 1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.1).
    """
    value = _acd.prevalence_from_apparent(pi, se, sp)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.1)"
    return RichResult(
        title='Prevalence from apparent probability, Se and Sp',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e1: pi_tilde = (pi + Sp - 1)/(Se + Sp - 1) [Bilder & Loughin 2025, eq. 6.1]'
