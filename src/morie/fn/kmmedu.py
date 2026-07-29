# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Medusa multi-head speculative decoding: K extra heads predict K
future tokens."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_medusa_heads"]


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kamath_medusa_heads(hidden_state, medusa_heads, k, verify=None):
    """p_k(y_{t+k}) = head_k(h_t), then accept/reject by verification.

    Each head is a callable ``h -> logits`` or a (d, V) matrix applied
    as ``h @ W``. Only the first ``k`` heads are used and the rest are
    reported as unused rather than quietly ignored. ``verify`` is the
    target model's checker, ``verify(position, token) -> bool``; the
    accepted prefix stops at the FIRST rejection, which is what makes
    speculative decoding lossless.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 10,
    Medusa heads; that section is not in the 2024 PDF, so the scheme
    is implemented exactly as the spec line states (Cai et al. 2024).

    Examples
    --------
    >>> h = [1.0, 0.0]
    >>> W1 = [[0.0, 1.0], [0.0, 0.0]]     # h @ W1 = [0, 1] -> token 1
    >>> W2 = [[1.0, 0.0], [0.0, 0.0]]     # h @ W2 = [1, 0] -> token 0
    >>> out = kamath_medusa_heads(h, [W1, W2], 2)
    >>> out["tokens"]
    [1, 0]
    >>> import math
    >>> abs(out["probabilities"][0] - math.e / (1 + math.e)) < 1e-12
    True
    >>> out2 = kamath_medusa_heads(h, [W1, W2], 2,
    ...                            verify=lambda i, t: i == 0)
    >>> out2["estimate"]
    1
    """
    h = np.atleast_1d(np.asarray(hidden_state, dtype=float)).ravel()
    k = int(k)
    heads = list(medusa_heads)
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}.")
    if k > len(heads):
        raise ValueError(
            f"asked for {k} speculative tokens but only {len(heads)} "
            "heads were supplied.")
    tokens, probs, all_probs = [], [], []
    for i, head in enumerate(heads[:k]):
        if callable(head):
            logits = np.atleast_1d(np.asarray(head(h), dtype=float)).ravel()
        else:
            W = np.atleast_2d(np.asarray(head, dtype=float))
            if W.shape[0] != h.size:
                raise ValueError(
                    f"head {i} is ({W.shape[0]}, {W.shape[1]}) but the "
                    f"hidden state has {h.size} dimensions.")
            logits = h @ W
        if logits.size < 2:
            raise ValueError(
                f"head {i} produced {logits.size} logits; a vocabulary "
                "of one token predicts nothing.")
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"head {i} produced a non-finite logit.")
        p = _softmax(logits)
        t = int(np.argmax(p))
        tokens.append(t)
        probs.append(float(p[t]))
        all_probs.append([float(v) for v in p])

    accepted = None
    if verify is not None:
        if not callable(verify):
            raise ValueError("verify must be callable (position, token) -> bool.")
        accepted = 0
        for i, t in enumerate(tokens):
            if not bool(verify(i, t)):
                break
            accepted += 1
    return RichResult(payload={
        "tokens": tokens, "probabilities": probs,
        "distributions": all_probs,
        "n_heads_available": len(heads), "n_heads_used": k,
        "accepted": accepted,
        "estimate": accepted if accepted is not None else probs[0],
        "n": k,
        "method": "Medusa multi-head speculative prediction"})


def cheatsheet():
    return "kmmedu: k heads -> k future tokens; accepted prefix stops at reject"
