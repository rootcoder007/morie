"""Cohen's d by the probit method d = probit(p1) - probit(p2).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["d_probit"]


def d_probit(p1, p2):
    """Cohen's d by the probit method d = probit(p1) - probit(p2)

    Formula: d = probit(p1) - probit(p2) = z1 - z2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.20
    """
    value = _ca_crim.d_probit(p1, p2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.20)"
    return RichResult(
        title="Cohen's d by the probit method d = probit(p1) - probit(p2)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e20: d = probit(p1) - probit(p2) = z1 - z2 [Weisburd et al. 2022, eq. 11.20]'


# compact alias per ledger/NAMING.md
dprobit = d_probit
