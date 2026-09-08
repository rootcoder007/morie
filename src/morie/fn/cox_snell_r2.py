"""Cox and Snell pseudo-R^2.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["cox_snell_r2"]


def cox_snell_r2(neg2ll_null, neg2ll_full, n):
    """Cox and Snell pseudo-R^2

    Formula: 1 - e^-[(-2LLnull) - (-2LLfull)]/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.4 eq.4.13
    """
    value = _ca_crim.cox_snell_r2(neg2ll_null, neg2ll_full, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (4.13)"
    return RichResult(
        title='Cox and Snell pseudo-R^2',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca4e13: 1 - e^-[(-2LLnull) - (-2LLfull)]/n [Weisburd et al. 2022, eq. 4.13]'


# compact alias per ledger/NAMING.md
coxsnellr2 = cox_snell_r2
