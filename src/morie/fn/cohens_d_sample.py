"""Cohen's d = (x1 - x2) / s_pooled.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["cohens_d_sample"]


def cohens_d_sample(m1, m2, s1, s2, n1, n2):
    """Cohen's d = (x1 - x2) / s_pooled

    Formula: d = (xbar_1 - xbar_2) / s_pooled

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.1
    """
    value = _ca_crim.cohens_d_sample(m1, m2, s1, s2, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.1)"
    return RichResult(
        title="Cohen's d = (x1 - x2) / s_pooled",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e1: d = (xbar_1 - xbar_2) / s_pooled [Weisburd et al. 2022, eq. 11.1]'


# compact alias per ledger/NAMING.md
cohensdsample = cohens_d_sample
