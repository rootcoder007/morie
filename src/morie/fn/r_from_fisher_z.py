"""Back-transform Zr to r.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["r_from_fisher_z"]


def r_from_fisher_z(z):
    """Back-transform Zr to r

    Formula: r = (e^{2Zr} - 1) / (e^{2Zr} + 1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.14
    """
    value = _ca_crim.r_from_fisher_z(z)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.14)"
    return RichResult(
        title='Back-transform Zr to r',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e14: r = (e^{2Zr} - 1) / (e^{2Zr} + 1) [Weisburd et al. 2022, eq. 11.14]'


# compact alias per ledger/NAMING.md
rfromfisherz = r_from_fisher_z
