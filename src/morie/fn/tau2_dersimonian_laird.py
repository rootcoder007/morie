"""DerSimonian-Laird tau^2 estimator.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["tau2_dersimonian_laird"]


def tau2_dersimonian_laird(ys, ws_fixed):
    """DerSimonian-Laird tau^2 estimator

    Formula: tau^2 = (Q - df) / (sum w - sum w^2 / sum w)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.44
    """
    value = _ca_crim.tau2_dersimonian_laird(ys, ws_fixed)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.44)"
    return RichResult(
        title='DerSimonian-Laird tau^2 estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e44: tau^2 = (Q - df) / (sum w - sum w^2 / sum w) [Weisburd et al. 2022, eq. 11.44]'
