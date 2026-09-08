"""Intraclass correlation rho = sigma2_u/(sigma2_u + sigma2_e).

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_7"]


def ca_chapter_7_equation_7(sigma2_u, sigma2_e):
    """Intraclass correlation rho = sigma2_u/(sigma2_u + sigma2_e)

    Formula: rho = sigma^2_u / (sigma^2_u + sigma^2_e)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.7
    """
    value = _ca_crim.intraclass_correlation(sigma2_u, sigma2_e)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.7)"
    return RichResult(
        title='Intraclass correlation rho = sigma2_u/(sigma2_u + sigma2_e)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e7: rho = sigma^2_u / (sigma^2_u + sigma^2_e) [Weisburd et al. 2022, eq. 7.7]'
