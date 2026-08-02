"""Loglinear odds ratio between rows i,i' and columns j,j'.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_7"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_7(bxz_ij, bxz_ipjp, bxz_ipj, bxz_ijp):
    """Loglinear odds ratio between rows i,i' and columns j,j'

    Formula: OR = exp(bXZ_ij + bXZ_i'j' - bXZ_i'j - bXZ_ij')

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.7).
    """
    value = _acd.loglinear_odds_ratio(bxz_ij, bxz_ipjp, bxz_ipj, bxz_ijp)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.7)"
    return RichResult(
        title="Loglinear odds ratio between rows i,i' and columns j,j'",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return "4e7: OR = exp(bXZ_ij + bXZ_i'j' - bXZ_i'j - bXZ_ij') [Bilder & Loughin 2025, eq. 4.7]"
