"""Cohen's d = (mu1 - mu2) / sigma.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["cohens_d_population"]


def cohens_d_population(mu1, mu2, sigma):
    """Cohen's d = (mu1 - mu2) / sigma

    Formula: d = (mu_1 - mu_2) / sigma

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.8 eq.8.2
    """
    value = _ca_crim.cohens_d_population(mu1, mu2, sigma)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (8.2)"
    return RichResult(
        title="Cohen's d = (mu1 - mu2) / sigma",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca8e2: d = (mu_1 - mu_2) / sigma [Weisburd et al. 2022, eq. 8.2]'
