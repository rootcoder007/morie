"""Pooled within-groups standard deviation.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["pooled_sd"]


def pooled_sd(s1, s2, n1, n2):
    """Pooled within-groups standard deviation

    Formula: s_pooled = sqrt(((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.2
    """
    value = _ca_crim.pooled_sd(s1, s2, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.2)"
    return RichResult(
        title='Pooled within-groups standard deviation',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e2: s_pooled = sqrt(((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2)) [Weisburd et al. 2022, eq. 11.2]'


# compact alias per ledger/NAMING.md
pooledsd = pooled_sd
