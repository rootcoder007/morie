# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.9: the round-trip (back-translation) prompt score."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_back_translation_prob"]


def _prob(v, name):
    v = float(v)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} = {v:.6g} is not a probability in [0, 1].")
    return v


def kamath_ch3_back_translation_prob(t, thatt, p_forward=None,
                                     p_backward=None):
    """P(t) = P_forward(t_hat|t) . P_backward(t|t_hat): the round-trip
    probability used to rank paraphrased prompt candidates.

    ``t`` is the candidate prompt, ``thatt`` (t_hat) its pivot-language
    translation; both are carried through for bookkeeping. The two leg
    probabilities must be supplied -- this function ranks, it does not
    translate.

    NOTE the printed equation reads "P_forward(t|t_hat) .
    P_backward(t|t)", which cannot be a round trip (the backward leg
    would not condition on the pivot). The coherent reading, and the
    one implemented, is forward t -> t_hat times backward t_hat -> t.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.9, printed
    p. 105.

    Examples
    --------
    >>> out = kamath_ch3_back_translation_prob(
    ...     "[z] is the capital of [x].", "[z] est la capitale de [x].",
    ...     p_forward=0.5, p_backward=0.25)
    >>> out["estimate"]
    0.125
    """
    if p_forward is None or p_backward is None:
        raise ValueError(
            "both p_forward and p_backward are required; Eq 3.9 is a "
            "product of two supplied leg probabilities.")
    pf = _prob(p_forward, "p_forward")
    pb = _prob(p_backward, "p_backward")
    return RichResult(payload={
        "estimate": pf * pb, "p_forward": pf, "p_backward": pb,
        "candidate": t, "pivot": thatt, "n": 2,
        "method": "round-trip back-translation score (Kamath Eq 3.9)"})


def cheatsheet():
    return "km050: P(t) = forward leg x backward leg, round-trip rank"
