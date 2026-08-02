# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 7.2: RAG answer-faithfulness metric."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch7_faithfulness_metric"]


def kamath_ch7_faithfulness_metric(facts):
    r"""Faithfulness = (# facts inferable from the context) / (# facts).

    ``facts`` is one indicator per atomic fact in the answer: 1/True
    when a judge (an LLM, in the book) says the retrieved context
    entails it, 0/False otherwise.

    This is the shared "supported fraction" core: ``km_fact`` (
    FactScore) and ``kmcrel`` (RAGAS context relevance) delegate here
    rather than repeat the ratio.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Eq 7.2, printed
    p. 300.

    Examples
    --------
    >>> out = kamath_ch7_faithfulness_metric([1, 1, 0, 1])
    >>> out["estimate"]
    0.75
    >>> out["n_supported"]
    3
    """
    f = np.atleast_1d(np.asarray(facts))
    if f.size == 0:
        raise ValueError("the answer contains no facts; faithfulness "
                         "is 0/0, which is undefined, not 1.")
    if f.dtype == bool:
        v = f.astype(float)
    else:
        v = f.astype(float)
        if not np.all((v == 0) | (v == 1)):
            raise ValueError("facts must be 0/1 (or boolean) support "
                             "indicators, one per atomic fact.")
    n_sup = int(v.sum())
    return RichResult(payload={
        "estimate": float(n_sup / v.size), "n_supported": n_sup,
        "n_facts": int(v.size), "n": int(v.size),
        "method": "RAG faithfulness = supported / total facts "
                  "(Kamath Eq 7.2)"})


def cheatsheet():
    return "km111: supported facts / total facts in the answer"
