"""One-multinomial contingency table PMF.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_2"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_2(count_table, prob_table):
    """One-multinomial contingency table PMF

    Formula: P(N11 = n11, ..., NIJ = nIJ) over I x J cells

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.2).
    """
    value = _acd.contingency_pmf(count_table, prob_table)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.2)"
    return RichResult(
        title='One-multinomial contingency table PMF',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e2: P(N11 = n11, ..., NIJ = nIJ) over I x J cells [Bilder & Loughin 2025, eq. 3.2]'
