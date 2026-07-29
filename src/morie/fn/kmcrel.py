# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 7: RAGAS context relevance."""

from ._richresult import RichResult
from .km111 import kamath_ch7_faithfulness_metric

__all__ = ["kamath_ragas_context_relevance"]


def kamath_ragas_context_relevance(context_sentences, relevance_labels):
    r"""ctxrel = |relevant sentences| / |sentences in the context|.

    One 0/1 label per retrieved context sentence. The ratio is the
    same supported-fraction core as Eq 7.2, so it is delegated to
    ``morie.fn.km111``; what is enforced here is that the labels line
    up with the sentences they judge.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Context Relevance;
    Es et al. (2023).

    Examples
    --------
    >>> out = kamath_ragas_context_relevance(["s1", "s2", "s3", "s4"],
    ...                                      [1, 0, 1, 1])
    >>> out["estimate"], out["n_relevant"]
    (0.75, 3)
    """
    sents = list(context_sentences)
    labs = list(relevance_labels)
    if len(sents) == 0:
        raise ValueError("the retrieved context has no sentences.")
    if len(sents) != len(labs):
        raise ValueError(
            f"{len(labs)} labels for {len(sents)} context sentences.")
    base = kamath_ch7_faithfulness_metric(labs)
    return RichResult(payload={
        "estimate": base["estimate"], "score": base["estimate"],
        "n_relevant": base["n_supported"],
        "n_sentences": len(sents), "n": len(sents),
        "method": "RAGAS context relevance (Kamath Ch 7; the ratio "
                  "core in km111)"})


def cheatsheet():
    return "kmcrel: relevant context sentences / retrieved sentences"
