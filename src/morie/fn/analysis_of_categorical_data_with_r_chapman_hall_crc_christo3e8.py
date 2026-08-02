"""One-at-a-time Wald interval for pi_j.

Book-as-spec implementation; see reference for context.
"""

import numpy as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_8"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_8(pi_hat, var_pi, z):
    """One-at-a-time Wald interval for pi_j

    Formula: pi_j_hat +/- z sqrt(Var(pi_j_hat))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.8).
    """
    payload = dict(_acd.pi_j_wald_interval(pi_hat, var_pi, z))
    value = float(payload['lower'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.8)"
    return RichResult(
        title='One-at-a-time Wald interval for pi_j',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e8: pi_j_hat +/- z sqrt(Var(pi_j_hat)) [Bilder & Loughin 2025, eq. 3.8]'
