"""Asymmetric power index.

Category: Spatial
"""

from . import _array_core as np
from ._containers import DescriptiveResult


def wvasym(weights_a, weights_b=None, quotas=None):
    """Asymmetric power index.

    Parameters
    ----------
    weights_a : array-like
        First set of weights.
    weights_b : array-like, optional
        Second set of weights.
    quotas : array-like, optional
        Quota thresholds.

    Returns
    -------
    DescriptiveResult
    """
    weights_a = np.asarray(weights_a, dtype=float)
    weights_b = np.ones_like(weights_a) if weights_b is None else np.asarray(weights_b, dtype=float)
    quotas = np.ones_like(weights_a) if quotas is None else np.asarray(quotas, dtype=float)
    combined = weights_a * weights_b
    total = float(np.sum(combined))
    stat = float(np.sum(combined * quotas) / total) if total > 0 else 0.0
    return DescriptiveResult(
        name="wvasym",
        value=stat,
        extra={"total_weight": total},
    )


short = "wvasym"
alias = "wvasym"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
wvasym = wvasym


def cheatsheet() -> str:
    return "wvasym({}) -> Asymmetric power index."
