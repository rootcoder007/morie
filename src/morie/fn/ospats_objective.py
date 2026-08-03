"""Ospats objective E_xi[O] (print eq 13.17).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["ospats_objective"]


def ospats_objective(per_stratum_sums, n_population):
    """Ospats objective E_xi[O] (print eq 13.17)

    Formula: E_xi[O] = (1/N) sum_h sqrt(sum_{i<j} E_xi[d2_ij])

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.17).
    """
    value = _brus.ospats_objective(per_stratum_sums, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.17)"
    return RichResult(
        title='Ospats objective E_xi[O] (print eq 13.17)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e17: E_xi[O] = (1/N) sum_h sqrt(sum_{i<j} E_xi[d2_ij]) [Brus 2022, eq. 13.17]'
