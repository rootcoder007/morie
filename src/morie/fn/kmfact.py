# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 6: FactScore, the supported-claim fraction."""

from ._richresult import RichResult
from .km111 import kamath_ch7_faithfulness_metric

__all__ = ["kamath_factscore"]


def kamath_factscore(atomic_claims, knowledge_base):
    r"""FactScore = |supported atomic claims| / |atomic claims|.

    ``knowledge_base`` is either a container of supported claims
    (membership decides support) or a callable
    ``knowledge_base(claim) -> bool``. The ratio itself is the same
    core as Eq 7.2, delegated to ``morie.fn.km111``; the support
    decision is what is done here, and the unsupported claims are
    listed so the score can be argued with.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, FactScore; Min et
    al. (2023).

    Examples
    --------
    >>> out = kamath_factscore(["a", "b"], {"a"})
    >>> out["estimate"], out["unsupported"]
    (0.5, ['b'])
    """
    claims = list(atomic_claims)
    if len(claims) == 0:
        raise ValueError("the generation was decomposed into no atomic "
                         "claims; FactScore is 0/0 there.")
    if callable(knowledge_base):
        flags = [1 if bool(knowledge_base(c)) else 0 for c in claims]
    else:
        try:
            flags = [1 if c in knowledge_base else 0 for c in claims]
        except TypeError:
            raise ValueError("knowledge_base must support `in` or be a "
                             "callable predicate.") from None
    base = kamath_ch7_faithfulness_metric(flags)
    return RichResult(payload={
        "estimate": base["estimate"], "score": base["estimate"],
        "supported": [c for c, f in zip(claims, flags) if f],
        "unsupported": [c for c, f in zip(claims, flags) if not f],
        "n_supported": base["n_supported"], "n": len(claims),
        "method": "FactScore (Kamath Ch 6; the ratio core in km111)"})


def cheatsheet():
    return "kmfact: share of atomic claims the knowledge base supports"
