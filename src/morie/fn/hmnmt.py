# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encoder-decoder neural machine translation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_encoder_decoder_nmt"]


def geron_encoder_decoder_nmt(src, tgt, model, max_len=None, eos=None):
    """
    Encoder-decoder for neural machine translation (seq2seq).

    Formula: enc(src) -> z; dec(z, tgt) -> y

    Two things happen here and they are NOT the same thing, which is the
    lesson of the chapter. Training runs with teacher forcing: the
    decoder is conditioned on the TRUE prefix at every step, so the loss
    is the sum of per-token cross-entropies against a prefix the model
    never had to produce. Inference has no such prefix and feeds back its
    own greedy output, so one early mistake shifts everything after it --
    exposure bias. Both are computed and returned, so the gap between the
    teacher-forced loss and the greedy decode is visible instead of
    assumed.

    ``model`` is a mapping of callables: ``encode(src) -> z`` and
    ``decode(z, prefix) -> probabilities`` over the vocabulary.

    Parameters
    ----------
    src : sequence
        Source token ids.
    tgt : sequence of int
        Target token ids (the reference).
    model : mapping
        ``{"encode": ..., "decode": ...}``.
    max_len : int, optional
        Greedy decode cap; defaults to ``len(tgt)``.
    eos : int, optional
        End-of-sequence id that stops the greedy decode.

    Returns
    -------
    result : RichResult
        Keys: loss, token_losses, perplexity, greedy, exact_match, z,
        estimate, n, method.

    Examples
    --------
    A model that always predicts the same distribution: every teacher-
    forced step costs -log 0.5 and the greedy decode always emits token 1.

    >>> model = {"encode": lambda s: len(s),
    ...          "decode": lambda z, prefix: [0.2, 0.5, 0.3]}
    >>> r = geron_encoder_decoder_nmt([9, 9], [1, 1], model)
    >>> [round(float(v), 6) for v in r["token_losses"]]
    [0.693147, 0.693147]
    >>> round(float(r["loss"]), 6), round(float(r["perplexity"]), 6)
    (1.386294, 2.0)
    >>> [int(t) for t in r["greedy"]], bool(r["exact_match"])
    ([1, 1], True)

    A reference the model does not favour costs more:

    >>> r2 = geron_encoder_decoder_nmt([9], [0, 0], model)
    >>> round(float(r2["loss"]), 6)
    3.218876

    References
    ----------
    Geron Ch 14
    """
    if not hasattr(model, "get"):
        raise ValueError("geron_encoder_decoder_nmt: model must be a mapping with 'encode' and 'decode' callables")
    encode, decode = model.get("encode"), model.get("decode")
    if not callable(encode) or not callable(decode):
        raise ValueError("geron_encoder_decoder_nmt: model needs callable 'encode' and 'decode'")
    s = list(src)
    t = [int(v) for v in tgt]
    if not s:
        raise ValueError("geron_encoder_decoder_nmt: src is empty")
    if not t:
        raise ValueError("geron_encoder_decoder_nmt: tgt is empty")

    z = encode(s)
    losses = []
    for i in range(len(t)):
        p = np.atleast_1d(np.asarray(decode(z, t[:i]), dtype=float)).ravel()
        if p.size == 0:
            raise ValueError(f"geron_encoder_decoder_nmt: decode returned no probabilities at step {i}")
        if np.any(p < 0) or not np.isclose(float(p.sum()), 1.0, atol=1e-6):
            raise ValueError(f"geron_encoder_decoder_nmt: decode must return a probability vector; step {i} sums to {p.sum()}")
        if not (0 <= t[i] < p.size):
            raise ValueError(f"geron_encoder_decoder_nmt: target token {t[i]} is outside the {p.size}-token vocabulary")
        losses.append(float(-np.log(max(float(p[t[i]]), 1e-300))))

    L = int(max_len) if max_len is not None else len(t)
    if L < 1:
        raise ValueError(f"geron_encoder_decoder_nmt: max_len must be >= 1, got {max_len!r}")
    greedy = []
    for _ in range(L):
        p = np.atleast_1d(np.asarray(decode(z, greedy), dtype=float)).ravel()
        nxt = int(np.argmax(p))
        greedy.append(nxt)
        if eos is not None and nxt == int(eos):
            break

    total = float(np.sum(losses))
    mean = total / len(t)
    return RichResult(
        title="Encoder-decoder NMT",
        summary_lines=[("Teacher-forced loss", total), ("Perplexity", float(np.exp(mean))), ("Greedy length", len(greedy))],
        interpretation="Teacher forcing conditions on the true prefix; greedy decoding does not, and that gap is exposure bias.",
        payload={
            "loss": total,
            "mean_loss": mean,
            "token_losses": losses,
            "perplexity": float(np.exp(mean)),
            "greedy": greedy,
            "exact_match": greedy[: len(t)] == t,
            "z": z,
            "estimate": total,
            "n": len(t),
            "method": "Teacher-forced cross-entropy plus greedy decoding through a supplied seq2seq model",
        },
    )


def cheatsheet():
    return "hmnmt: Encoder-decoder NMT loss and greedy decode"
