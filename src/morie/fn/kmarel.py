# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 7: RAGAS answer relevance from reverse-generated questions."""

import numpy as np

from ._richresult import RichResult
from .km112 import kamath_ch7_answer_relevance

__all__ = ["kamath_ragas_answer_relevance"]


def kamath_ragas_answer_relevance(answer, original_question, model,
                                  embed=None):
    r"""rel = (1/n) sum_i cos(embed(q_i_reverse), embed(q_orig)).

    ``model(answer)`` is the caller's LLM step: it must return the
    reverse-generated questions' EMBEDDINGS (n x d), or their text
    together with an ``embed`` callable that turns text into vectors.
    Once the embeddings exist the metric is Eq 7.3 exactly, so the
    mean cosine is delegated to ``morie.fn.km112``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, Answer Relevance
    (Eq 7.3); Es et al. (2023).

    Examples
    --------
    >>> gen = lambda a: [[1.0, 0.0], [0.0, 1.0]]
    >>> out = kamath_ragas_answer_relevance("cats purr", [1.0, 0.0], gen)
    >>> out["estimate"]
    0.5
    """
    if not callable(model):
        raise ValueError("model must be callable model(answer) -> the "
                         "reverse-generated questions.")
    qs = model(answer)
    if qs is None or len(list(qs)) == 0:
        raise ValueError("the model generated no reverse questions; "
                         "answer relevance is a mean over none.")
    qs = list(qs)
    if embed is not None:
        if not callable(embed):
            raise ValueError("embed must be callable or None.")
        E_g = np.array([np.asarray(embed(q), dtype=float).ravel()
                        for q in qs])
        E_o = np.asarray(embed(original_question), dtype=float).ravel()
    else:
        E_g = np.atleast_2d(np.asarray(qs, dtype=float))
        E_o = np.asarray(original_question, dtype=float).ravel()
    base = kamath_ch7_answer_relevance(E_g, E_o)
    return RichResult(payload={
        "estimate": base["estimate"], "score": base["estimate"],
        "similarities": base["similarities"], "n": base["n"],
        "method": "RAGAS answer relevance (Kamath Ch 7; the Eq 7.3 "
                  "core in km112)"})


def cheatsheet():
    return "kmarel: mean cosine of reverse-generated questions to the query"
