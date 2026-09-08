# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 8: BERTScore end to end, from tokens to P/R/F1."""

from . import _array_core as np

from ._richresult import RichResult
from .km119 import kamath_ch8_bertscore_recall
from .km120 import kamath_ch8_bertscore_precision
from .km121 import kamath_ch8_bertscore_f1

__all__ = ["kamath_bertscore"]


def kamath_bertscore(hypothesis_tokens, reference_tokens, embed_fn):
    r"""P, R and F1 from cosine similarities of contextual embeddings.

    ``embed_fn`` is the caller's contextual encoder: a callable
    applied to the whole token list (returning one row per token) or
    to a single token, or a mapping token -> vector. The three
    formulas are Eqs 8.7-8.9 and live in ``morie.fn.km119``, ``km120``
    and ``km121``; embeddings are cosine-normalized before they are
    used, as BERTScore specifies.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, BERTScore;
    Zhang et al. (2020).

    Examples
    --------
    >>> emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    >>> out = kamath_bertscore(["a"], ["a", "b"], emb)
    >>> out["precision"], out["recall"]
    (1.0, 0.5)
    >>> round(out["f1"], 6)
    0.666667
    """
    hyp = list(hypothesis_tokens)
    ref = list(reference_tokens)
    if len(hyp) == 0 or len(ref) == 0:
        raise ValueError("both token sequences must be non-empty.")

    def embed(tokens):
        if hasattr(embed_fn, "get") and not callable(embed_fn):
            missing = [t for t in tokens if t not in embed_fn]
            if missing:
                raise ValueError(f"no embedding for {missing!r}.")
            M = np.array([np.asarray(embed_fn[t], dtype=float).ravel()
                          for t in tokens])
        else:
            if not callable(embed_fn):
                raise ValueError("embed_fn must be callable or a "
                                 "token -> vector mapping.")
            try:
                M = np.atleast_2d(np.asarray(embed_fn(tokens),
                                             dtype=float))
                if M.shape[0] != len(tokens):
                    raise ValueError
            except Exception:
                M = np.array([np.asarray(embed_fn(t),
                                         dtype=float).ravel()
                              for t in tokens])
        if M.shape[0] != len(tokens):
            raise ValueError("embed_fn returned "
                             f"{M.shape[0]} vectors for "
                             f"{len(tokens)} tokens.")
        return M

    H = embed(hyp)
    R = embed(ref)
    rec = kamath_ch8_bertscore_recall(R, H, normalize=True)
    pre = kamath_ch8_bertscore_precision(R, H, normalize=True)
    f1 = kamath_ch8_bertscore_f1(pre["estimate"], rec["estimate"])
    return RichResult(payload={
        "estimate": f1["estimate"], "f1": f1["estimate"],
        "precision": pre["estimate"], "recall": rec["estimate"],
        "candidate_match": pre["greedy_match"],
        "reference_match": rec["greedy_match"],
        "n": len(hyp),
        "method": "BERTScore (Kamath Ch 8; km119/km120/km121 cores)"})


def cheatsheet():
    return "kmbsco: embed both token lists, then km119/km120/km121"
