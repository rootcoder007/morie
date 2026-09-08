"""Law of total probability: P(Z) = sum_i P(Z | Ai) P(Ai).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (2.29), (2.55), (2.86).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["ptotal"]


def ptotal(priors, likelihoods):
    """P(Z) = sum_i P(Z | Ai) P(Ai) over a complete partition.

    Parameters
    ----------
    priors : array-like
        Prior probabilities of a complete, mutually exclusive set; must
        sum to 1.
    likelihoods : array-like
        P(Z | Ai), same length as ``priors``.

    Returns
    -------
    RichResult
        Keys: p_total, and the aliases p_event / p_z / p_b kept for the
        eq (2.29) / (2.55) / (2.86) callers.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (2.29), (2.55), (2.86).
    """
    value = _morin.total_probability(priors, likelihoods)
    payload = {"p_total": value, "p_event": value, "p_z": value, "p_b": value}
    lines = [("P(Z)", value)]
    return RichResult(
        title="Law of total probability: P(Z) = sum P(Z|Ai) P(Ai).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "ptotal: P(Z) = sum P(Z|Ai) P(Ai). Morin (2016) eqs (2.29), (2.55), (2.86)."
