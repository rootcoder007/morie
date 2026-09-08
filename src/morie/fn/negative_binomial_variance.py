"""Negative binomial variance Var(Y) = mu + mu^2 alpha.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["negative_binomial_variance"]


def negative_binomial_variance(mu, alpha):
    """Negative binomial variance Var(Y) = mu + mu^2 alpha

    Formula: Var(Y) = mu + mu^2 alpha

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.6 eq.6.8
    """
    value = _ca_crim.negative_binomial_variance(mu, alpha)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (6.8)"
    return RichResult(
        title='Negative binomial variance Var(Y) = mu + mu^2 alpha',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca6e8: Var(Y) = mu + mu^2 alpha [Weisburd et al. 2022, eq. 6.8]'
