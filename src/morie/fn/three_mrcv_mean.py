"""Complete independence for three MRCVs.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["three_mrcv_mean"]


def three_mrcv_mean(b0, beta_w_a, beta_y_b, beta_z_c):
    """Complete independence for three MRCVs

    Formula: log(mu_abc(ijk)) = b0 + bW_a + bY_b + bZ_c

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.16).
    """
    value = _acd.three_mrcv_mean(b0, beta_w_a, beta_y_b, beta_z_c)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.16)"
    return RichResult(
        title='Complete independence for three MRCVs',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e16: log(mu_abc(ijk)) = b0 + bW_a + bY_b + bZ_c [Bilder & Loughin 2025, eq. 6.16]'


# compact alias per ledger/NAMING.md
threemrcvmean = three_mrcv_mean
