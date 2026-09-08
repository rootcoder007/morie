"""Logit with a two-categorical-variable interaction.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["interaction_logit"]


def interaction_logit(b, x1, x2, z1, z2):
    """Logit with a two-categorical-variable interaction

    Formula: logit(pi) = b0 + b1 x1 + b2 x2 + b3 z1 + b4 z2 + b5 x1 z1 + ... + b8 x2 z2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.22).
    """
    value = _acd.interaction_logit(b, x1, x2, z1, z2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.22)"
    return RichResult(
        title='Logit with a two-categorical-variable interaction',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e22: logit(pi) = b0 + b1 x1 + b2 x2 + b3 z1 + b4 z2 + b5 x1 z1 + ... + b8 x2 z2 [Bilder & Loughin 2025, eq. 2.22]'
