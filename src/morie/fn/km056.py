# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.3: the full-parameter fine-tuning objective."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_full_finetune_obj"]


def _sequence_objective(model, x, y):
    """sum over pairs, sum over target positions, of log p(y_t|x,y_<t).

    ``model`` is a callable (x_i, y_prefix, y_t) -> probability in
    (0, 1]. km057 imports this so the LoRA objective and the full
    fine-tuning objective are the SAME sum, differing only in which
    parameter set produced the probabilities.
    """
    xs = list(x)
    ys = [list(seq) for seq in y]
    if not xs:
        raise ValueError("Z is empty; a sum over no context-target pairs "
                         "is undefined, not 0.")
    if len(xs) != len(ys):
        raise ValueError(
            f"got {len(xs)} contexts for {len(ys)} targets.")
    if not callable(model):
        raise ValueError("the model must be a callable (x, y_prefix, y_t) "
                         "-> probability.")
    per_pair, total = [], 0.0
    for xi, yi in zip(xs, ys):
        if not yi:
            raise ValueError("a target sequence is empty; log p over no "
                             "tokens is undefined.")
        s = 0.0
        for t, tok in enumerate(yi):
            p = float(model(xi, yi[:t], tok))
            if not (0.0 < p <= 1.0):
                raise ValueError(
                    f"the model returned {p:.6g}; probabilities must lie "
                    "in (0, 1].")
            s += math.log(p)
        per_pair.append(s)
        total += s
    return total, per_pair


def kamath_ch4_full_finetune_obj(Phi, x, y):
    """max_Phi sum_{(x,y) in Z} sum_t log P_Phi(y_t | x, y_<t).

    The objective every parameter of Phi is updated against. ``Phi`` is
    the caller's model as a callable (x_i, y_prefix, y_t) ->
    probability; ``x`` the contexts, ``y`` the target sequences. The
    value returned is the objective, so LARGER is better.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.3, printed
    p. 150.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch4_full_finetune_obj(
    ...     lambda xi, pre, t: 0.5, ["doc"], [["a", "b"]])
    >>> abs(out["estimate"] - 2 * math.log(0.5)) < 1e-12
    True
    >>> out["n_tokens"]
    2
    """
    total, per_pair = _sequence_objective(Phi, x, y)
    return RichResult(payload={
        "estimate": float(total), "per_pair": per_pair,
        "n_tokens": int(sum(len(list(s)) for s in y)), "n": len(per_pair),
        "method": "full-parameter fine-tuning objective (Kamath Eq 4.3)"})


def cheatsheet():
    return "km056: sum_pairs sum_t log P_Phi(y_t|x,y_<t), maximise"
