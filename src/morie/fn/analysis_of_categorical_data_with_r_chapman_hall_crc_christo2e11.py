"""Wald confidence interval for OR = exp(c beta1).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_11"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_11(b1, var_b1, c, z):
    """Wald confidence interval for OR = exp(c beta1)

    Formula: exp(c b1 +/- c z sqrt(Var(b1)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'or' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.11).
    """
    payload = dict(_acd.or_ci_logistic(b1, var_b1, c, z))
    value = float(payload['or'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.11)"
    return RichResult(
        title='Wald confidence interval for OR = exp(c beta1)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e11: exp(c b1 +/- c z sqrt(Var(b1))) [Bilder & Loughin 2025, eq. 2.11]'
