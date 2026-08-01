"""Cohen's d from an independent t-test.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_11_equation_5"]


def ca_chapter_11_equation_5(t, n1, n2):
    """Cohen's d from an independent t-test

    Formula: d = t sqrt((n1+n2)/(n1 n2))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.11 eq.11.5
    """
    value = _ca_crim.d_from_t(t, n1, n2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (11.5)"
    return RichResult(
        title="Cohen's d from an independent t-test",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca11e5: d = t sqrt((n1+n2)/(n1 n2)) [Weisburd et al. 2022, eq. 11.5]'
