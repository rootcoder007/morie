"""Wald interval for an ordinal-association odds ratio.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_50"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_50(b1, var_b1, c, z):
    """Wald interval for an ordinal-association odds ratio

    Formula: exp(c b +/- c z sqrt(Var(b))) (provenance: the '3.50' in the stub name is a printed OR confidence bound, not an equation)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'or' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.50).
    """
    payload = dict(_acd.or_ci_logistic(b1, var_b1, c, z))
    value = float(payload['or'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.50)"
    return RichResult(
        title='Wald interval for an ordinal-association odds ratio',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return "3e50: exp(c b +/- c z sqrt(Var(b))) (provenance: the '3.50' in the stub name is a printed OR confidence bound, not an equation) [Bilder & Loughin 2025, eq. 3.50]"
