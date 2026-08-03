"""Poisson regression log-likelihood.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["poisson_loglik"]


def poisson_loglik(b, x, y):
    """Poisson regression log-likelihood

    Formula: sum -exp(Xb_i) + y_i Xb_i - log(y_i!)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.3).
    """
    value = _acd.poisson_loglik(b, x, y)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.3)"
    return RichResult(
        title='Poisson regression log-likelihood',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e3: sum -exp(Xb_i) + y_i Xb_i - log(y_i!) [Bilder & Loughin 2025, eq. 4.3]'
