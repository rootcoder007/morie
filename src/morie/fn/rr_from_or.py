"""Convert an odds ratio to a risk ratio.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["rr_from_or"]


def rr_from_or(or_value, p2):
    """Convert an odds ratio to a risk ratio

    Formula: RR = OR / (1 - p2 + p2 OR)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.29
    """
    value = _ca_crim.rr_from_or(or_value, p2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.29)"
    return RichResult(
        title='Convert an odds ratio to a risk ratio',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e29: RR = OR / (1 - p2 + p2 OR) [Weisburd et al. 2022, eq. 11.29]'


# compact alias per ledger/NAMING.md
rrfromor = rr_from_or
