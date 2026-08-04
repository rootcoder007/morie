"""Wilson (score) confidence interval for pi.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["wilson_interval"]


def wilson_interval(w, n, z):
    """Wilson (score) confidence interval for pi

    Formula: pi_tilde +/- (z sqrt(n)/(n+z^2)) sqrt(pi_hat(1-pi_hat) + z^2/4n)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'estimate' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.4).
    """
    payload = dict(_acd.wilson_interval(w, n, z))
    value = float(payload['estimate'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.4)"
    return RichResult(
        title='Wilson (score) confidence interval for pi',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e4: pi_tilde +/- (z sqrt(n)/(n+z^2)) sqrt(pi_hat(1-pi_hat) + z^2/4n) [Bilder & Loughin 2025, eq. 1.4]'


# compact alias per ledger/NAMING.md
wilsoninterval = wilson_interval
