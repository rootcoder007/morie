"""Delta-method variance of a survey proportion.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_9"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_9(var_ni, var_n, cov_ni_n, pi_hat, n_hat):
    """Delta-method variance of a survey proportion

    Formula: Var(pi_i) = (Var(N_i) + pi^2 Var(N) - 2 pi Cov(N_i, N))/N^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.9).
    """
    value = _acd.survey_proportion_variance(var_ni, var_n, cov_ni_n, pi_hat, n_hat)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.9)"
    return RichResult(
        title='Delta-method variance of a survey proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e9: Var(pi_i) = (Var(N_i) + pi^2 Var(N) - 2 pi Cov(N_i, N))/N^2 [Bilder & Loughin 2025, eq. 6.9]'
