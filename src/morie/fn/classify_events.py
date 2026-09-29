"""Classify an event pair: independent and/or exclusive.

Implements eq (2.24) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["classify_events"]


def classify_events(p_a, p_b, p_ab, tol=1e-12):
    """Classify an event pair: independent and/or exclusive.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.24).

    Examples
    --------
    >>> classify_events(0.5, 0.4, 0.2)["independent"]
    True
    """
    independent, exclusive = _morin.classify_events(p_a, p_b, p_ab, tol)
    payload = {
        "independent": independent,
        "exclusive": exclusive,
        "p_a": float(p_a),
        "p_b": float(p_b),
        "p_ab": float(p_ab),
    }
    lines = [("independent", independent), ("exclusive", exclusive)]
    return RichResult(
        title="Classify an event pair: independent and/or exclusive.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e24: Classify an event pair: independent and/or exclusive. Morin (2016) eq (2.24)."


# compact alias per ledger/NAMING.md
classifyevents = classify_events
