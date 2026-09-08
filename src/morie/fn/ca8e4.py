"""Cohen's f = sigma_means / sigma_error.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_8_equation_4"]


def ca_chapter_8_equation_4(sigma_means, sigma_error):
    """Cohen's f = sigma_means / sigma_error

    Formula: f = sigma_m / sigma_e

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.4
    """
    value = _ca_crim.cohens_f(sigma_means, sigma_error)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.4)"
    return RichResult(
        title="Cohen's f = sigma_means / sigma_error",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e4: f = sigma_m / sigma_e [Weisburd et al. 2022, eq. 8.4]'
