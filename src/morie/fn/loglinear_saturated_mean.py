"""Saturated loglinear model with interaction.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["loglinear_saturated_mean"]


def loglinear_saturated_mean(b0, beta_x_i, beta_z_j, beta_xz_ij):
    """Saturated loglinear model with interaction

    Formula: log(mu_ij) = b0 + bX_i + bZ_j + bXZ_ij

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.6).
    """
    value = _acd.loglinear_saturated_mean(b0, beta_x_i, beta_z_j, beta_xz_ij)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.6)"
    return RichResult(
        title='Saturated loglinear model with interaction',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e6: log(mu_ij) = b0 + bX_i + bZ_j + bXZ_ij [Bilder & Loughin 2025, eq. 4.6]'
