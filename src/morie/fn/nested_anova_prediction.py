"""Nested ANOVA model for variogram sampling.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["nested_anova_prediction"]


def nested_anova_prediction(mu, a_i, b_ij, c_ijk, eps):
    """Nested ANOVA model for variogram sampling

    Formula: Z_ijkl = mu + A_i + B_ij + C_ijk + eps_ijkl

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (24.1).
    """
    value = _brus.nested_anova_prediction(mu, a_i, b_ij, c_ijk, eps)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (24.1)"
    return RichResult(
        title='Nested ANOVA model for variogram sampling',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r24e1: Z_ijkl = mu + A_i + B_ij + C_ijk + eps_ijkl [Brus 2022, eq. 24.1]'
