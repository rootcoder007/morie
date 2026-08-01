"""Dependent-samples (paired) t-test.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_9_equation_10"]


def ca_chapter_9_equation_10(differences):
    """Dependent-samples (paired) t-test

    Formula: t = dbar / sqrt(s_d^2 / n); df = n_pairs - 1

    Returns
    -------
    result : RichResult
        dict subclass; headline key 't' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.9 eq.9.10
    """
    payload = dict(_ca_crim.t_paired(differences))
    value = payload['t']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (9.10)"
    return RichResult(
        title='Dependent-samples (paired) t-test',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca9e10: t = dbar / sqrt(s_d^2 / n); df = n_pairs - 1 [Weisburd et al. 2022, eq. 9.10]'
