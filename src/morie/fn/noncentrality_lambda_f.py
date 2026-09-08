"""Noncentrality lambda = n f^2 for the F distribution.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["noncentrality_lambda_f"]


def noncentrality_lambda_f(f, n_total):
    """Noncentrality lambda = n f^2 for the F distribution

    Formula: lambda = n f^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.5
    """
    value = _ca_crim.noncentrality_lambda_f(f, n_total)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.5)"
    return RichResult(
        title='Noncentrality lambda = n f^2 for the F distribution',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e5: lambda = n f^2 [Weisburd et al. 2022, eq. 8.5]'
