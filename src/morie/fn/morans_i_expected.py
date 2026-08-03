"""Expected Moran's I under no autocorrelation E(I) = -1/(n-1).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["morans_i_expected"]


def morans_i_expected(n):
    """Expected Moran's I under no autocorrelation E(I) = -1/(n-1)

    Formula: E(I) = -1 / (n - 1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.12 eq.12.2
    """
    value = _ca_crim.morans_i_expected(n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (12.2)"
    return RichResult(
        title="Expected Moran's I under no autocorrelation E(I) = -1/(n-1)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca12e2: E(I) = -1 / (n - 1) [Weisburd et al. 2022, eq. 12.2]'
