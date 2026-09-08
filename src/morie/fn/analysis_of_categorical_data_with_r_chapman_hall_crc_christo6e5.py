"""Exact conditional distribution over permutations.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_5"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_5(t_values, counts, beta, t_obs):
    """Exact conditional distribution over permutations

    Formula: P(Y|I) = exp(bp sum y x)/sum_R exp(bp sum y* x)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'p_at_t' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.5).
    """
    payload = dict(_acd.exact_conditional_pmf(t_values, counts, beta, t_obs))
    value = float(payload['p_at_t'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.5)"
    return RichResult(
        title='Exact conditional distribution over permutations',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e5: P(Y|I) = exp(bp sum y x)/sum_R exp(bp sum y* x) [Bilder & Loughin 2025, eq. 6.5]'
