"""Poisson regression log link log(mu) = Xb.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["poisson_log_link"]


def poisson_log_link(b0, bs, xs):
    """Poisson regression log link log(mu) = Xb

    Formula: log(mu) = b0 + b1 x1 + ... + bp xp

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.2).
    """
    value = _acd.poisson_log_link(b0, bs, xs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.2)"
    return RichResult(
        title='Poisson regression log link log(mu) = Xb',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e2: log(mu) = b0 + b1 x1 + ... + bp xp [Bilder & Loughin 2025, eq. 4.2]'


# compact alias per ledger/NAMING.md
poissonloglink = poisson_log_link
