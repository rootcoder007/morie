"""Bernoulli likelihood L = prod pi^y (1-pi)^(1-y) (log form).

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_1"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_1(pis, ys):
    """Bernoulli likelihood L = prod pi^y (1-pi)^(1-y) (log form)

    Formula: L(pi_1..pi_n|y) = prod pi_i^y_i (1-pi_i)^(1-y_i)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.1).
    """
    value = _acd.bernoulli_likelihood(pis, ys)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.1)"
    return RichResult(
        title='Bernoulli likelihood L = prod pi^y (1-pi)^(1-y) (log form)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e1: L(pi_1..pi_n|y) = prod pi_i^y_i (1-pi_i)^(1-y_i) [Bilder & Loughin 2025, eq. 2.1]'
