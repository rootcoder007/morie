"""Overall model F-test from R^2.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["f_overall_r2"]


def f_overall_r2(r2, n, k):
    """Overall model F-test from R^2

    Formula: F = R^2 (n-k-1) / ((1-R^2) k)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.2 eq.2.17
    """
    value = _ca_crim.f_overall_r2(r2, n, k)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (2.17)"
    return RichResult(
        title='Overall model F-test from R^2',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca2e17: F = R^2 (n-k-1) / ((1-R^2) k) [Weisburd et al. 2022, eq. 2.17]'


# compact alias per ledger/NAMING.md
foverallr2 = f_overall_r2
