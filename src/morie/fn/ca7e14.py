"""Level-1 model of the random coefficient decomposition.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_14"]


def ca_chapter_7_equation_14(beta_0j, beta_1j, x_1ij):
    """Level-1 model of the random coefficient decomposition

    Formula: y_ij = beta_0j + beta_1j x_1ij + e_ij

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.14
    """
    value = _ca_crim.linear_predictor(beta_0j, [beta_1j], [x_1ij])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.14)"
    return RichResult(
        title='Level-1 model of the random coefficient decomposition',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e14: y_ij = beta_0j + beta_1j x_1ij + e_ij [Weisburd et al. 2022, eq. 7.14]'
