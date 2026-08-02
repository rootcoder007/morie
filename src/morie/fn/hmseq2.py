# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sequence-to-sequence encoder-decoder architecture."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsftm import geron_softmax_function

__all__ = ["geron_seq2seq"]


def geron_seq2seq(src, tgt, encoder, decoder, max_len=None, eos=None):
    """
    Sequence-to-sequence encoder-decoder architecture.

    Formula: enc(x_1..x_T) -> z; dec(z, y_0..y_{s-1}) -> y_s

    Orchestrates the two caller-supplied halves and enforces their
    contract: ``encoder(src)`` returns a fixed-length context vector `z`,
    and ``decoder(z, prefix)`` returns a vocabulary-sized score vector for
    the *next* token given the tokens produced so far. Two passes are run
    over the same decoder: teacher forcing (prefix taken from `tgt`),
    which gives the training cross-entropy
    ``-mean_s log softmax(dec(z, y_<s))[y_s]``, and free-running greedy
    decoding, which gives the inference output. The gap between them is
    exposure bias, and it is reported rather than hidden.

    Parameters
    ----------
    src : array-like
        Source sequence; passed to `encoder` unchanged.
    tgt : array-like of int
        Target token ids used for teacher forcing.
    encoder : callable
        ``encoder(src) -> z`` (1-D array-like, non-empty, finite).
    decoder : callable
        ``decoder(z, prefix) -> scores`` of constant length V >= 2.
    max_len : int, optional
        Greedy decode length; defaults to ``len(tgt)``.
    eos : int, optional
        Token id that stops greedy decoding.

    Returns
    -------
    result : RichResult
        Keys: z, loss, perplexity, greedy, token_logprobs, exposure_bias,
        estimate, n, method.

    Examples
    --------
    A decoder that scores every token equally gives the uniform loss
    log V and decodes token 0 at every step:

    >>> import numpy as np
    >>> z_of = lambda s: np.asarray([float(sum(s))])
    >>> dec = lambda z, prefix: np.zeros(3)
    >>> r = geron_seq2seq([1, 2], [1, 2], z_of, dec)
    >>> round(float(r["loss"]), 6)
    1.098612
    >>> round(float(r["perplexity"]), 6)
    3.0
    >>> [int(v) for v in r["greedy"]]
    [0, 0]

    A decoder that copies the source length picks a definite token, so
    the loss on that token is (almost) zero:

    >>> dec2 = lambda z, prefix: np.asarray([0.0, 20.0, 0.0])
    >>> r2 = geron_seq2seq([1, 2], [1, 1], z_of, dec2)
    >>> bool(r2["loss"] < 1e-8)
    True
    >>> [int(v) for v in r2["greedy"]]
    [1, 1]

    References
    ----------
    Géron Ch 13
    """
    if not callable(encoder) or not callable(decoder):
        raise ValueError("geron_seq2seq: encoder and decoder must both be callables")
    y = np.asarray(tgt).ravel()
    if y.size == 0:
        raise ValueError("geron_seq2seq: tgt is empty; there is nothing to teacher-force")
    if not np.all(np.equal(np.mod(y.astype(float), 1), 0)):
        raise ValueError("geron_seq2seq: tgt must contain integer token ids")
    y = y.astype(int)

    z = np.asarray(encoder(src), dtype=float).ravel()
    if z.size == 0:
        raise ValueError("geron_seq2seq: encoder returned an empty context vector")
    if not np.all(np.isfinite(z)):
        raise ValueError("geron_seq2seq: encoder returned non-finite values")

    def _scores(prefix, where):
        s = np.asarray(decoder(z, list(prefix)), dtype=float).ravel()
        if s.size < 2:
            raise ValueError(f"geron_seq2seq: decoder returned {s.size} scores at {where}; need a vocabulary of >= 2")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"geron_seq2seq: decoder returned non-finite scores at {where}")
        return s

    first = _scores([], "step 0")
    V = first.size
    if y.min() < 0 or y.max() >= V:
        raise ValueError(f"geron_seq2seq: tgt ids must lie in 0..{V - 1}, got {int(y.min())}..{int(y.max())}")

    logps = np.empty(y.size)
    for s in range(y.size):
        sc = first if s == 0 else _scores(y[:s], f"step {s}")
        if sc.size != V:
            raise ValueError(f"geron_seq2seq: decoder changed vocabulary size from {V} to {sc.size} at step {s}")
        p = np.asarray(geron_softmax_function(sc)["p"], dtype=float)
        logps[s] = np.log(max(float(p[y[s]]), np.finfo(float).tiny))
    loss = float(-np.mean(logps))

    L = int(y.size) if max_len is None else int(max_len)
    if L < 1:
        raise ValueError(f"geron_seq2seq: max_len must be >= 1, got {L}")
    greedy = []
    greedy_lp = 0.0
    for s in range(L):
        sc = first if s == 0 else _scores(greedy, f"greedy step {s}")
        p = np.asarray(geron_softmax_function(sc)["p"], dtype=float)
        k = int(np.argmax(sc))
        greedy_lp += float(np.log(max(float(p[k]), np.finfo(float).tiny)))
        greedy.append(k)
        if eos is not None and k == int(eos):
            break

    matched = int(sum(1 for a, b in zip(greedy, y.tolist()) if a == b))

    return RichResult(
        title="Sequence-to-sequence",
        summary_lines=[
            ("Target length", int(y.size)),
            ("Vocabulary", int(V)),
            ("Teacher-forced loss", loss),
            ("Greedy matches", matched),
        ],
        interpretation=(
            "The whole source is squeezed into one context vector, so long inputs lose detail -- "
            "this is the bottleneck attention was invented to remove."
        ),
        payload={
            "z": z,
            "loss": loss,
            "perplexity": float(np.exp(loss)),
            "token_logprobs": logps,
            "greedy": np.asarray(greedy, dtype=int),
            "greedy_logprob": greedy_lp,
            "greedy_matches": matched,
            "exposure_bias": float(1.0 - matched / max(1, min(len(greedy), y.size))),
            "vocab_size": int(V),
            "estimate": loss,
            "n": int(y.size),
            "method": "Encoder-decoder: teacher-forced cross-entropy plus greedy free-running decode",
        },
    )


def cheatsheet():
    return "hmseq2: Sequence-to-sequence encoder-decoder architecture"
