# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instruction-tuning cross-entropy over the response tokens only."""

import numpy as np

from ._richresult import RichResult
from .kmclm import kamath_causal_lm_loss

__all__ = ["kamath_instruction_tuning_loss"]

_IGNORE = -100


def kamath_instruction_tuning_loss(logits, response_mask, targets):
    """L = -(1/|R|) sum_{t in R} log p(r_t | instruction, r_{<t}).

    Instruction tuning is next-token cross-entropy with the
    instruction positions masked OUT of the average, so this DELEGATES
    to ``morie.fn.kmclm`` (the causal-LM loss) after replacing the
    non-response targets with its ignore index -- one implementation
    of one formula, no drift.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, instruction tuning.

    Examples
    --------
    >>> import math
    >>> lg = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    >>> out = kamath_instruction_tuning_loss(lg, [0, 1, 1], [0, 1, 0])
    >>> abs(out["estimate"] - math.log(2)) < 1e-12
    True
    >>> out["n_response_tokens"]
    2
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if logits.ndim != 2:
        raise ValueError("logits must be (T, V) or (B, T, V).")
    m = np.asarray(response_mask).ravel()
    t = np.asarray(targets).ravel()
    if m.size != logits.shape[0]:
        raise ValueError(
            f"response_mask has {m.size} entries for {logits.shape[0]} "
            "positions.")
    if t.size != logits.shape[0]:
        raise ValueError(
            f"targets has {t.size} entries for {logits.shape[0]} "
            "positions.")
    if m.dtype != bool and not np.all(np.isin(m, (0, 1))):
        raise ValueError("response_mask must be boolean or 0/1.")
    keep = m.astype(bool)
    if not keep.any():
        raise ValueError(
            "the response mask selects no token; instruction tuning "
            "with nothing to imitate has no loss.")
    masked_targets = np.where(keep, t.astype(int), _IGNORE)
    base = kamath_causal_lm_loss(logits, masked_targets,
                                 ignore_index=_IGNORE)
    return RichResult(payload={
        "estimate": float(base["loss"]),
        "loss": float(base["loss"]),
        "perplexity": float(base["perplexity"]),
        "token_losses": [float(v) for v in base["token_losses"]],
        "n_response_tokens": int(base["n_tokens"]),
        "vocab_size": int(base["vocab_size"]),
        "n": int(logits.shape[0]),
        "method": "Instruction-tuning CE over response tokens "
                  "(delegates to kmclm)"})


def cheatsheet():
    return "kminst: kmclm's causal CE with the instruction masked out"
