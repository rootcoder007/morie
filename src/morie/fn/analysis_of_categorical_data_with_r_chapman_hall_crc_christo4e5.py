"""Loglinear independence cell mean log(mu_ij) = b0 + bX_i + bZ_j.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_5"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_5(b0, beta_x_i, beta_z_j):
    """Loglinear independence cell mean log(mu_ij) = b0 + bX_i + bZ_j

    Formula: log(mu_ij) = b0 + bX_i + bZ_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.5).
    """
    value = _acd.loglinear_independence_mean(b0, beta_x_i, beta_z_j)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.5)"
    return RichResult(
        title='Loglinear independence cell mean log(mu_ij) = b0 + bX_i + bZ_j',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e5: log(mu_ij) = b0 + bX_i + bZ_j [Bilder & Loughin 2025, eq. 4.5]'
