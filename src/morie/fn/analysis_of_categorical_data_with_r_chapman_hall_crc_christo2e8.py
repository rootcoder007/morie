"""Residual deviance of a null model vs the saturated model.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_8"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_8(pis, ys):
    """Residual deviance of a null model vs the saturated model

    Formula: -2 sum y log(pi/y) + (1-y) log((1-pi)/(1-y))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.8).
    """
    value = _acd.residual_deviance(pis, ys)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.8)"
    return RichResult(
        title='Residual deviance of a null model vs the saturated model',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e8: -2 sum y log(pi/y) + (1-y) log((1-pi)/(1-y)) [Bilder & Loughin 2025, eq. 2.8]'
