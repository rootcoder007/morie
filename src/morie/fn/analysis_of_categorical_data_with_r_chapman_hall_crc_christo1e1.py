"""Binomial PMF P(W = w) = C(n, w) pi^w (1-pi)^(n-w).

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_1"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_1_equation_1(w, n, p):
    """Binomial PMF P(W = w) = C(n, w) pi^w (1-pi)^(n-w)

    Formula: P(W = w) = C(n, w) pi^w (1-pi)^(n-w)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.1).
    """
    value = _acd.binomial_pmf(w, n, p)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.1)"
    return RichResult(
        title='Binomial PMF P(W = w) = C(n, w) pi^w (1-pi)^(n-w)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e1: P(W = w) = C(n, w) pi^w (1-pi)^(n-w) [Bilder & Loughin 2025, eq. 1.1]'
