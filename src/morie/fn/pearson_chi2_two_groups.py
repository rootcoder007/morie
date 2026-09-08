"""Two-group Pearson chi-square with pooled proportion.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["pearson_chi2_two_groups"]


def pearson_chi2_two_groups(w1, n1, w2, n2):
    """Two-group Pearson chi-square with pooled proportion

    Formula: X2 = sum_j (w_j - n_j pibar)^2/(n_j pibar) + (failures term)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'x2' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.7).
    """
    payload = dict(_acd.pearson_chi2_two_groups(w1, n1, w2, n2))
    value = float(payload['x2'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.7)"
    return RichResult(
        title='Two-group Pearson chi-square with pooled proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e7: X2 = sum_j (w_j - n_j pibar)^2/(n_j pibar) + (failures term) [Bilder & Loughin 2025, eq. 1.7]'
