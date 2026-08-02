"""True confidence level C(pi) of a binomial interval.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_6"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_6(n, p, interval_fn):
    """True confidence level C(pi) of a binomial interval

    Formula: C(pi) = sum_w I(w) C(n,w) pi^w (1-pi)^(n-w)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.6).
    """
    value = _acd.true_confidence_level(n, p, interval_fn)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.6)"
    return RichResult(
        title='True confidence level C(pi) of a binomial interval',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e6: C(pi) = sum_w I(w) C(n,w) pi^w (1-pi)^(n-w) [Bilder & Loughin 2025, eq. 1.6]'
