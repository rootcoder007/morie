"""Wald confidence interval for the odds ratio.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["or_wald_interval"]


def or_wald_interval(w1, n1, w2, n2, z):
    """Wald confidence interval for the odds ratio

    Formula: exp(log(OR_hat) +/- z sqrt(1/w1 + 1/(n1-w1) + 1/w2 + 1/(n2-w2)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'or' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.10).
    """
    payload = dict(_acd.or_wald_interval(w1, n1, w2, n2, z))
    value = float(payload['or'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.10)"
    return RichResult(
        title='Wald confidence interval for the odds ratio',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e10: exp(log(OR_hat) +/- z sqrt(1/w1 + 1/(n1-w1) + 1/w2 + 1/(n2-w2))) [Bilder & Loughin 2025, eq. 1.10]'


# compact alias per ledger/NAMING.md
orwaldinterval = or_wald_interval
