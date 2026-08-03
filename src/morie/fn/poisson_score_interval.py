"""Score confidence interval for a Poisson mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["poisson_score_interval"]


def poisson_score_interval(mu_hat, n, z):
    """Score confidence interval for a Poisson mean

    Formula: mu_hat + z^2/2n +/- z sqrt((mu_hat + z^2/4n)/n)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (4.1).
    """
    payload = dict(_acd.poisson_score_interval(mu_hat, n, z))
    value = float(payload['lower'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (4.1)"
    return RichResult(
        title='Score confidence interval for a Poisson mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '4e1: mu_hat + z^2/2n +/- z sqrt((mu_hat + z^2/4n)/n) [Bilder & Loughin 2025, eq. 4.1]'
