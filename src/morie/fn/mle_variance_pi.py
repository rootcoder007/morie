"""MLE variance Var(pi_hat) = pi_hat(1 - pi_hat)/n.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["mle_variance_pi"]


def mle_variance_pi(pi_hat, n):
    """MLE variance Var(pi_hat) = pi_hat(1 - pi_hat)/n

    Formula: Var(pi_hat) = pi_hat(1-pi_hat)/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (1.3).
    """
    value = _acd.mle_variance_pi(pi_hat, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (1.3)"
    return RichResult(
        title='MLE variance Var(pi_hat) = pi_hat(1 - pi_hat)/n',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '1e3: Var(pi_hat) = pi_hat(1-pi_hat)/n [Bilder & Loughin 2025, eq. 1.3]'


# compact alias per ledger/NAMING.md
mlevariancepi = mle_variance_pi
