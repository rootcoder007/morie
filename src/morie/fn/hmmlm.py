# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Masked language modeling pretraining objective."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_masked_lm"]

_METHOD = "Masked language modelling objective"


def geron_masked_lm(X, mask_frac=0.15, model=None, vocab_size=None, seed=0, mask_token=-1):
    """
    Masked language modeling pretraining objective.

    Formula: L = -sum_{i in M} log P(x_i | x_{not M})

    BERT's objective.  The loss is computed **only over the masked
    positions** -- averaging over the whole sequence instead would let a
    model score well by copying the unmasked tokens it can already see,
    which is the trivial solution the masking exists to forbid.

    Bidirectionality is the payoff and the constraint: the prediction at
    a masked position conditions on tokens to its right as well as its
    left, which is why an MLM cannot be used to generate left-to-right.

    The default ``model`` is the unigram distribution estimated from the
    *unmasked* tokens with add-one smoothing -- a genuine, computable
    baseline that conditions on the visible context in the weakest way
    possible.  Any real model should beat it, and its loss is returned
    as ``baseline_loss`` for that comparison.  Supply
    ``model(masked_sequence, positions) -> (n_masked, vocab)`` of
    probabilities to score a real one; the shape and row sums are
    enforced.

    Parameters
    ----------
    X : array-like of int
        Token id sequence, or a batch of sequences of equal length.
    mask_frac : float
        Fraction of positions masked, in (0, 1).
    model : callable, optional
        ``model(masked_X, positions) -> probability matrix``.
    vocab_size : int, optional
        Vocabulary size; inferred as ``max(X) + 1`` if omitted.
    seed : int
        Seed for choosing mask positions.
    mask_token : int
        Value substituted at masked positions.

    Returns
    -------
    result : RichResult
        Keys: loss, baseline_loss, perplexity, masked_positions,
        targets, probabilities, n_masked, estimate, n, method.

    Examples
    --------
    A sequence of 20 tokens with 15% masking loses 3 positions:

    >>> X = [0, 1, 2, 3] * 5
    >>> r = geron_masked_lm(X, mask_frac=0.15, seed=0)
    >>> r["n_masked"]
    3

    A model that is certain and correct has zero loss:

    >>> perfect = lambda mx, pos: np.eye(4)[[X[p] for p in pos]]
    >>> p = geron_masked_lm(X, mask_frac=0.15, seed=0, model=perfect, vocab_size=4)
    >>> round(p["loss"], 12)
    0.0
    >>> round(p["perplexity"], 12)
    1.0

    A uniform model over 4 tokens costs ``log 4`` per position:

    >>> uniform = lambda mx, pos: np.full((len(pos), 4), 0.25)
    >>> u = geron_masked_lm(X, mask_frac=0.15, seed=0, model=uniform, vocab_size=4)
    >>> round(u["loss"], 9)
    1.386294361
    >>> round(u["perplexity"], 6)
    4.0

    With no model the loss is the smoothed-unigram baseline by
    construction:

    >>> round(r["loss"], 12) == round(r["baseline_loss"], 12)
    True

    A model returning rows that are not distributions is refused:

    >>> bad = lambda mx, pos: np.full((len(pos), 4), 0.5)
    >>> geron_masked_lm(X, mask_frac=0.15, seed=0, model=bad, vocab_size=4)
    Traceback (most recent call last):
        ...
    ValueError: geron_masked_lm: model returned rows that do not sum to 1 (row 0 sums to 2)

    References
    ----------
    Géron Ch 15
    """
    A = np.asarray(X)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_masked_lm: X must be a non-empty sequence or batch of sequences, got shape {A.shape}")
    if not np.issubdtype(A.dtype, np.integer):
        f = np.asarray(A, dtype=float)
        if not np.all(f == np.floor(f)):
            raise ValueError("geron_masked_lm: X must contain integer token ids")
        A = f.astype(np.int64)
    if np.any(A < 0):
        raise ValueError("geron_masked_lm: token ids must be non-negative")
    V = int(A.max()) + 1 if vocab_size is None else int(vocab_size)
    if V < 1 or np.any(A >= V):
        raise ValueError(f"geron_masked_lm: token ids must lie in 0..{V - 1}, got max {int(A.max())}")
    frac = float(mask_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError(f"geron_masked_lm: mask_frac must lie in (0, 1), got {mask_frac!r}")

    flat = A.ravel()
    n_tok = flat.size
    n_mask = max(1, int(round(frac * n_tok)))
    if n_mask >= n_tok:
        raise ValueError(
            f"geron_masked_lm: masking {n_mask} of {n_tok} tokens leaves no visible context"
        )
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(n_tok, size=n_mask, replace=False))
    targets = flat[positions]

    masked = flat.copy()
    masked[positions] = int(mask_token)
    masked_view = masked.reshape(A.shape)

    visible = np.delete(flat, positions)
    counts = np.bincount(visible, minlength=V).astype(float) + 1.0
    unigram = counts / counts.sum()
    baseline_probs = np.tile(unigram, (n_mask, 1))
    baseline_loss = float(np.mean(-np.log(baseline_probs[np.arange(n_mask), targets])))

    if model is None:
        probs = baseline_probs
    else:
        if not callable(model):
            raise ValueError(f"geron_masked_lm: model must be callable, got {type(model).__name__}")
        probs = np.atleast_2d(np.asarray(model(masked_view, positions), dtype=float))
        if probs.shape != (n_mask, V):
            raise ValueError(
                f"geron_masked_lm: model returned shape {probs.shape}, expected ({n_mask}, {V})"
            )
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("geron_masked_lm: model returned negative or non-finite probabilities")
        sums = probs.sum(axis=1)
        bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-6)
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"geron_masked_lm: model returned rows that do not sum to 1 (row {i} sums to {sums[i]:g})"
            )

    picked = probs[np.arange(n_mask), targets]
    loss = float(np.mean(-np.log(np.clip(picked, 1e-300, None))))
    ppl = float(np.exp(loss))

    return RichResult(
        title="Masked language modelling",
        summary_lines=[
            ("Tokens", int(n_tok)),
            ("Masked", int(n_mask)),
            ("Loss (nats/token)", loss),
            ("Perplexity", ppl),
            ("Smoothed-unigram baseline", baseline_loss),
        ],
        interpretation=(
            "Scored only at the masked positions, so copying the visible tokens earns nothing; "
            "conditioning is bidirectional, which is why an MLM cannot generate left to right."
        ),
        payload={
            "loss": loss,
            "baseline_loss": baseline_loss,
            "perplexity": ppl,
            "masked_positions": positions,
            "masked_input": masked_view,
            "targets": targets,
            "probabilities": probs,
            "n_masked": int(n_mask),
            "vocab_size": V,
            "estimate": loss,
            "n": int(n_tok),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmlm: MLM loss over masked positions only, against a smoothed-unigram baseline"
