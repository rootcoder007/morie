# morie.fn -- function file (rootcoder007/morie)
"""CBOW: predict the centre word from averaged context embeddings."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_cbow"]


def burkov_cbow(context_ids, center_ids, embeddings, output_weights,
                output_bias=None):
    r"""Continuous bag-of-words forward pass and loss.

    .. math::
       h = \frac{1}{|C|}\sum_{c \in C} v_c, \qquad
       p(w \mid C) = \mathrm{softmax}(W h + b)

    The averaging is what the name means and what the model gives up:
    order is discarded entirely, so "dog bites man" and "man bites dog"
    produce the same hidden vector. CBOW accepts that in exchange for a
    single forward pass per window regardless of context size, which is
    why it trains faster than skip-gram.

    The two differ in what they learn, not only in speed. CBOW averages
    the context, so a rare word's contribution is diluted by its
    frequent neighbours and its embedding is learned poorly; skip-gram
    predicts each context word separately and therefore gives rare
    words their own gradient signal. ``rare_word_dilution`` reports
    :math:`1/|C|`, the factor by which any single context word's
    contribution is scaled down.

    Parameters
    ----------
    context_ids : array-like, shape (n, k)
        Context word indices per example.
    center_ids : array-like, shape (n,)
    embeddings : array-like, shape (V, d)
    output_weights : array-like, shape (V, d)
    output_bias : array-like, shape (V,), optional

    Returns
    -------
    RichResult
        ``loss``, ``hidden``, ``logits``, ``probabilities``,
        ``predicted``, ``accuracy``, ``perplexity``,
        ``rare_word_dilution``.

    References
    ----------
    Burkov (2025), *The Language Model Book*, chapter 2, CBOW.
    Mikolov, Chen, Corrado and Dean (2013), "Efficient estimation of
    word representations in vector space", arXiv:1301.3781.

    Examples
    --------
    >>> import numpy as np
    >>> E = np.eye(3)
    >>> out = burkov_cbow([[0, 1]], [2], E, np.eye(3))
    >>> out["hidden"].shape
    (1, 3)
    """
    C = np.atleast_2d(np.asarray(context_ids, dtype=int))
    y = np.asarray(center_ids, dtype=int).ravel()
    E = np.atleast_2d(np.asarray(embeddings, dtype=float))
    W = np.atleast_2d(np.asarray(output_weights, dtype=float))
    V, d = E.shape
    if W.shape[1] != d:
        raise ValueError(
            "output_weights has dimension %d, embeddings %d."
            % (W.shape[1], d)
        )
    n, k = C.shape
    if y.size != n:
        raise ValueError(
            "center_ids has %d entries for %d context rows." % (y.size, n)
        )
    if C.min() < 0 or C.max() >= V or y.min() < 0 or y.max() >= W.shape[0]:
        raise ValueError("a word index is out of range for the vocabulary.")
    b = np.zeros(W.shape[0]) if output_bias is None else np.asarray(
        output_bias, dtype=float
    ).ravel()
    if b.size != W.shape[0]:
        raise ValueError(
            "output_bias has %d entries for %d output rows."
            % (b.size, W.shape[0])
        )

    h = E[C].mean(axis=1)
    logits = h @ W.T + b
    mx = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - mx)
    prob = ex / ex.sum(axis=1, keepdims=True)
    pick = np.clip(prob[np.arange(n), y], 1e-300, None)
    loss = float(-np.mean(np.log(pick)))
    pred = np.argmax(prob, axis=1)
    return RichResult(
        payload={
            "estimate": loss,
            "loss": loss,
            "hidden": h,
            "logits": logits,
            "probabilities": prob,
            "predicted": pred,
            "accuracy": float(np.mean(pred == y)),
            "perplexity": float(np.exp(loss)),
            "rare_word_dilution": float(1.0 / k),
            "dilution_note": (
                "each context word contributes 1/|C| of the hidden vector, "
                "so a rare word is drowned out by frequent neighbours; "
                "skip-gram gives every context word its own gradient instead"
            ),
            "order_note": (
                "averaging discards word order entirely -- any permutation "
                "of the context gives the same hidden vector"
            ),
            "context_size": int(k),
            "vocab_size": int(V),
            "dim": int(d),
            "n": int(n),
            "method": "CBOW forward pass and cross-entropy loss",
        }
    )


def cheatsheet():
    return (
        "bkcbow: CBOW averaged-context softmax, with the order loss and "
        "rare-word dilution it trades for speed"
    )
