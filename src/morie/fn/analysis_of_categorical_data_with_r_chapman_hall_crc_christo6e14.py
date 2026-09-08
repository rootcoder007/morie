"""Independence loglinear model for one item pair.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_14"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_14(b0, beta_w_a, beta_y_b):
    """Independence loglinear model for one item pair

    Formula: log(mu_ab) = b0 + bW_a + bY_b

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.14).
    """
    value = _acd.spmi_loglinear_mean(b0, beta_w_a, beta_y_b)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.14)"
    return RichResult(
        title='Independence loglinear model for one item pair',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e14: log(mu_ab) = b0 + bW_a + bY_b [Bilder & Loughin 2025, eq. 6.14]'
