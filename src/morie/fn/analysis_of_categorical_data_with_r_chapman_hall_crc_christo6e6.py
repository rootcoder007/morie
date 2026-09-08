"""Exact conditional PMF of the sufficient statistic.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_6"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_6(t_values, counts, beta, t_obs):
    """Exact conditional PMF of the sufficient statistic

    Formula: P(T = t_u|I) = c(t_u) exp(bp t_u)/sum_v c(t_v) exp(bp t_v)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'p_at_t' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.6).
    """
    payload = dict(_acd.exact_conditional_pmf(t_values, counts, beta, t_obs))
    value = float(payload['p_at_t'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.6)"
    return RichResult(
        title='Exact conditional PMF of the sufficient statistic',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e6: P(T = t_u|I) = c(t_u) exp(bp t_u)/sum_v c(t_v) exp(bp t_v) [Bilder & Loughin 2025, eq. 6.6]'
