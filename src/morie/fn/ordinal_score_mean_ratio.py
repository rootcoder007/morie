"""Ordinal-score mean ratio between column levels.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["ordinal_score_mean_ratio"]


def ordinal_score_mean_ratio(beta_z_j, beta_z_jp, beta_xz_i, s_j, s_jp):
    """Ordinal-score mean ratio between column levels

    Formula: mu_ij/mu_ij' = exp((bZ_j - bZ_j') + bXZ_i (s_j - s_j'))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.12).
    """
    value = _acd.ordinal_score_mean_ratio(beta_z_j, beta_z_jp, beta_xz_i, s_j, s_jp)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.12)"
    return RichResult(
        title='Ordinal-score mean ratio between column levels',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return "4e12: mu_ij/mu_ij' = exp((bZ_j - bZ_j') + bXZ_i (s_j - s_j')) [Bilder & Loughin 2025, eq. 4.12]"
