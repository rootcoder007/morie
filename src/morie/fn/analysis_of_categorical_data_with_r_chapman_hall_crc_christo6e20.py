"""Logistic GLMM for clustered binary falls data.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_20"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_20(b0, bs, xs, random_intercept):
    """Logistic GLMM for clustered binary falls data

    Formula: logit(pi_ik) = b0 + b2 x2 + b3 x3 + b4 x4 + b_0i

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.20).
    """
    value = _acd.glmm_linear_predictor(b0, 1.0, float(np.dot(np.atleast_1d(np.asarray(bs, dtype=float)), np.atleast_1d(np.asarray(xs, dtype=float)))), random_intercept)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.20)"
    return RichResult(
        title='Logistic GLMM for clustered binary falls data',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e20: logit(pi_ik) = b0 + b2 x2 + b3 x3 + b4 x4 + b_0i [Bilder & Loughin 2025, eq. 6.20]'
