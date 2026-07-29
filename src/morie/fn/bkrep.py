# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 5: the repetition penalty on decoder logits."""

import numpy as np

from ._richresult import RichResult

__all__ = ["burkov_repetition_penalty"]


def burkov_repetition_penalty(logits, prev_tokens, penalty=1.2):
    """Divide positive logits of seen tokens by the penalty, multiply
    negative ones -- both move the token DOWN in probability, which is
    why the sign split exists (dividing a negative logit would move it
    UP).

    References: Burkov LM (2025), Ch 5, repetition penalty (the CTRL
    rule of Keskar et al. 2019).

    Examples
    --------
    >>> burkov_repetition_penalty([2.0, -2.0, 1.0], [0, 1], 2.0)["penalised"]
    [1.0, -4.0, 1.0]
    """
    z = np.atleast_1d(np.asarray(logits, dtype=float)).copy()
    r = float(penalty)
    if r <= 0:
        raise ValueError(f"penalty must be positive; got {penalty}.")
    prev = sorted({int(t) for t in np.atleast_1d(
        np.asarray(prev_tokens)).astype(int)})
    for t in prev:
        if not 0 <= t < len(z):
            raise ValueError(
                f"token index {t} is out of range for {len(z)} logits.")
        z[t] = z[t] / r if z[t] > 0 else z[t] * r
    return RichResult(payload={
        "penalised": [float(v) for v in z], "estimate": float(z[0]),
        "penalty": r, "tokens_hit": prev, "n": len(z),
        "method": "Repetition penalty on logits (Burkov Ch 5)"})


def cheatsheet():
    return "bkrep: repetition penalty, sign-split divide/multiply (Burkov Ch 5)"
