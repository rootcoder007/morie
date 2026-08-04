"""Proportion of variance R^2 = f^2 / (1 + f^2).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["r2_from_f2"]


def r2_from_f2(f2):
    """Proportion of variance R^2 = f^2 / (1 + f^2)

    Formula: R^2 = f^2 / (1 + f^2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.7
    """
    value = _ca_crim.r2_from_f2(f2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.7)"
    return RichResult(
        title='Proportion of variance R^2 = f^2 / (1 + f^2)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e7: R^2 = f^2 / (1 + f^2) [Weisburd et al. 2022, eq. 8.7]'


# compact alias per ledger/NAMING.md
r2fromf2 = r2_from_f2
