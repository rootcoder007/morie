"""Design-effect-adjusted sample size.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["n_design_effect"]


def n_design_effect(design_effect, n_si):
    """Design-effect-adjusted sample size

    Formula: n(p, zbar) = sqrt(de) n(SI, pi)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.14).
    """
    value = _brus.n_design_effect(design_effect, n_si)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.14)"
    return RichResult(
        title='Design-effect-adjusted sample size',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e14: n(p, zbar) = sqrt(de) n(SI, pi) [Brus 2022, eq. 12.14]'
