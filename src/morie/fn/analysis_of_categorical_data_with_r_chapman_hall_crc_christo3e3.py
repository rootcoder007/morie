"""Product multinomial PMF (I independent row multinomials).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_3(count_table, cond_prob_table):
    """Product multinomial PMF (I independent row multinomials)

    Formula: prod_i [n_i+!/(prod_j n_ij!) prod_j pi_j|i^n_ij]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.3).
    """
    value = _acd.product_multinomial_pmf(count_table, cond_prob_table)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.3)"
    return RichResult(
        title='Product multinomial PMF (I independent row multinomials)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e3: prod_i [n_i+!/(prod_j n_ij!) prod_j pi_j|i^n_ij] [Bilder & Loughin 2025, eq. 3.3]'
