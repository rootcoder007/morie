# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.2: the softmax over label words' pre-softmax vectors."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_prompt_softmax_label"]


def kamath_ch3_prompt_softmax_label(w, h_z, M):
    """p(y|x) = exp(w_{M(y)} . h_z) / sum_{y'} exp(w_{M(y')} . h_z).

    ``w`` maps an answer word to its pre-softmax vector, ``h_z`` is the
    hidden vector of the answer slot, ``M`` maps labels to answer
    words. The normaliser runs over the LABEL words only (the book's
    y' in y), not the whole vocabulary.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.2, printed
    p. 91.

    Examples
    --------
    >>> w = {"great": [1.0, 0.0], "terrible": [0.0, 1.0]}
    >>> M = {"pos": "great", "neg": "terrible"}
    >>> out = kamath_ch3_prompt_softmax_label(w, [1.0, 0.0], M)
    >>> round(out["label_probs"]["pos"], 10)   # 1/(1+exp(-1))
    0.7310585786
    >>> out["label"]
    'pos'
    """
    if not isinstance(M, dict) or not M:
        raise ValueError("M must be a non-empty label -> answer-word map.")
    if not isinstance(w, dict) or not w:
        raise ValueError("w must be a non-empty answer-word -> vector map.")
    h = np.atleast_1d(np.asarray(h_z, dtype=float))
    if h.size == 0:
        raise ValueError("h_z is empty.")
    labels = list(M.keys())
    logits = []
    for lab in labels:
        word = M[lab]
        if word not in w:
            raise ValueError(
                f"answer word {word!r} (label {lab!r}) has no vector in w.")
        v = np.atleast_1d(np.asarray(w[word], dtype=float))
        if v.shape != h.shape:
            raise ValueError(
                f"w[{word!r}] has shape {v.shape}, h_z has {h.shape}.")
        logits.append(float(v @ h))
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - z.max())
    p = e / e.sum()
    best = int(np.argmax(p))
    return RichResult(payload={
        "estimate": float(p[best]), "label": labels[best],
        "label_probs": {lab: float(pi) for lab, pi in zip(labels, p)},
        "logits": {lab: float(zi) for lab, zi in zip(labels, z)},
        "n": len(labels),
        "method": "label-word softmax (Kamath Eq 3.2)"})


def cheatsheet():
    return "km043: softmax of w_{M(y)}.h_z over the label words only"
