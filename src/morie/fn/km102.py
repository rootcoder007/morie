# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.26: the chain rule for a word sequence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_lstm_chain_rule"]


def kamath_ch6_lstm_chain_rule(w_1_w_M):
    """P(w_1,...,w_M) = prod_{t=1..M} P(w_t | w_1,...,w_{t-1}).

    ``w_1_w_M`` holds the per-step CONDITIONAL probabilities, one per
    position. The product underflows to 0 for long sequences, so the
    log probability (the sum of logs, computed independently) is
    returned alongside and is the number to compare across sequences.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.26, printed
    p. 252.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_lstm_chain_rule([0.5, 0.25])
    >>> out["estimate"]
    0.125
    >>> abs(out["log_prob"] + math.log(8.0)) < 1e-12
    True
    """
    p = np.atleast_1d(np.asarray(w_1_w_M, dtype=float))
    if p.size == 0:
        raise ValueError("the sequence is empty; a product over no tokens "
                         "is not a sentence probability.")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("every conditional probability must lie in "
                         "(0, 1]; a zero makes the log probability "
                         "undefined.")
    return RichResult(payload={
        "estimate": float(np.prod(p)),
        "log_prob": float(np.sum(np.log(p))),
        "per_step": [float(v) for v in p], "n": int(p.size),
        "method": "chain rule sequence probability (Kamath Eq 6.26)"})


def cheatsheet():
    return "km102: product of per-step conditionals, plus its log"
