"""Loglinear independence model (indicator form).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_4"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_4(b0, beta_x_i, beta_z_j):
    """Loglinear independence model (indicator form)

    Formula: log(mu) = b0 + bX_2 x2 + ... + bZ_2 z2 + ...

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.4).
    """
    value = _acd.loglinear_independence_mean(b0, beta_x_i, beta_z_j)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.4)"
    return RichResult(
        title='Loglinear independence model (indicator form)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e4: log(mu) = b0 + bX_2 x2 + ... + bZ_2 z2 + ... [Bilder & Loughin 2025, eq. 4.4]'
