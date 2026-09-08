# morie.fn -- function file (rootcoder007/morie)
"""Sequential (causally ordered) mediators."""

from ._richresult import RichResult
from .medstg import sequential_mediation

__all__ = ["sequential_mediators"]


def sequential_mediators(y, x, m1, m2, c=None):
    """Causally ordered mediators M1 -> M2; front-end to :mod:`medstg`.

    Same serial decomposition as
    :func:`morie.fn.medstg.sequential_mediation`, with the
    outcome-first argument order used elsewhere in the mediation
    namespace.

    References
    ----------
    Hayes, A. F. (2022). *Introduction to Mediation, Moderation, and
    Conditional Process Analysis* (3rd ed.). Guilford Press. Ch. 5.
    """
    out = sequential_mediation(x, m1, m2, y, c=c)
    payload = dict(out)
    payload["method"] = "Sequential (causally ordered) mediators M1 -> M2"
    return RichResult(payload=payload)


def cheatsheet():
    return "seqM: causally ordered mediators (medstg front-end, y-first signature)"
