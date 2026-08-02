"""Posterior for regression parameters (grid-normalized).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_25"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_25(logliks, log_priors):
    """Posterior for regression parameters (grid-normalized)

    Formula: p(beta|y) prop prod f(y_i|beta) prod p(beta_r)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.25).
    """
    value = float(_acd.posterior_kernel_regression(logliks, log_priors)[0])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.25)"
    return RichResult(
        title='Posterior for regression parameters (grid-normalized)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e25: p(beta|y) prop prod f(y_i|beta) prod p(beta_r) [Bilder & Loughin 2025, eq. 6.25]'
