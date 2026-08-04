"""Chi-square for a 2x2 frequency table.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["chi2_2x2"]


def chi2_2x2(a, b, c, d):
    """Chi-square for a 2x2 frequency table

    Formula: chi2 = (ad-bc)^2 (a+b+c+d) / [(a+b)(c+d)(a+c)(b+d)]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'chi2' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.4
    """
    payload = dict(_ca_crim.chi2_2x2(a, b, c, d))
    value = payload['chi2']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.4)"
    return RichResult(
        title='Chi-square for a 2x2 frequency table',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e4: chi2 = (ad-bc)^2 (a+b+c+d) / [(a+b)(c+d)(a+c)(b+d)] [Weisburd et al. 2022, eq. 9.4]'


# compact alias per ledger/NAMING.md
chi22x2 = chi2_2x2
