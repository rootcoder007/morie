"""Hedges' small-sample correction J = 1 - 3/(4(n1+n2)-9).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["hedges_j"]


def hedges_j(n1, n2):
    """Hedges' small-sample correction J = 1 - 3/(4(n1+n2)-9)

    Formula: J = 1 - 3/(4(n1+n2)-9)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.3
    """
    value = _ca_crim.hedges_j(n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.3)"
    return RichResult(
        title="Hedges' small-sample correction J = 1 - 3/(4(n1+n2)-9)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e3: J = 1 - 3/(4(n1+n2)-9) [Weisburd et al. 2022, eq. 11.3]'


# compact alias per ledger/NAMING.md
hedgesj = hedges_j
