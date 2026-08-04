"""Multinomial PMF.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["multinomial_pmf"]


def multinomial_pmf(counts, probs):
    """Multinomial PMF

    Formula: P(N1 = n1, ..., NJ = nJ) = n!/(prod n_j!) prod pi_j^n_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (3.1).
    """
    value = _acd.multinomial_pmf(counts, probs)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (3.1)"
    return RichResult(
        title='Multinomial PMF',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '3e1: P(N1 = n1, ..., NJ = nJ) = n!/(prod n_j!) prod pi_j^n_j [Bilder & Loughin 2025, eq. 3.1]'


# compact alias per ledger/NAMING.md
multinomialpmf = multinomial_pmf
