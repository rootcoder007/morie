"""Kott-Carr effective-sample-size interval for a proportion.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["kott_carr_interval"]


def kott_carr_interval(pi_hat, var_pi, t_crit):
    """Kott-Carr effective-sample-size interval for a proportion

    Formula: Wilson-type interval with n* = pi(1-pi)/Var(pi) and t critical

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.11).
    """
    payload = dict(_acd.kott_carr_interval(pi_hat, var_pi, t_crit))
    value = float(payload['lower'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.11)"
    return RichResult(
        title='Kott-Carr effective-sample-size interval for a proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e11: Wilson-type interval with n* = pi(1-pi)/Var(pi) and t critical [Bilder & Loughin 2025, eq. 6.11]'
