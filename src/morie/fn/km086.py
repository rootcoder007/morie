# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.10: pseudo-log-likelihood (PLL)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_pll"]


def _log_probs(items, scorer, name):
    """Per-token log probabilities, from raw probabilities or a scorer.

    ``scorer`` is a callable index -> probability; when it is None the
    items must already BE probabilities. km087 and km088 import this.
    """
    seq = list(items)
    if not seq:
        raise ValueError(f"{name} is empty; a sum over no tokens is "
                         "undefined, not 0.")
    logs = []
    for i, tok in enumerate(seq):
        p = float(scorer(i)) if scorer is not None else float(tok)
        if not (0.0 < p <= 1.0):
            raise ValueError(
                f"the conditional probability at position {i} is "
                f"{p:.6g}; it must lie in (0, 1].")
        logs.append(np.log(p))
    return np.asarray(logs, dtype=float), seq


def kamath_ch6_pll(S, theta=None):
    """PLL(S) = sum_{s in S} log P(s | S_without_s ; theta).

    Mask one token at a time and read off its probability given every
    OTHER token -- a bidirectional model's stand-in for a likelihood.
    ``theta`` is a callable index -> P(s_i | S\\s_i); with theta = None,
    ``S`` is taken to be those probabilities already. Always <= 0, and
    longer sentences score lower, which is why CrowS-Pairs compares
    minimal pairs rather than raw PLLs.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.10, printed
    p. 235.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_pll([0.5, 0.25])
    >>> abs(out["estimate"] + math.log(8.0)) < 1e-12
    True
    >>> kamath_ch6_pll(["a", "b"], theta=lambda i: 1.0)["estimate"]
    0.0
    """
    if theta is not None and not callable(theta):
        raise ValueError("theta must be a callable index -> probability, "
                         "or None when S already holds probabilities.")
    logs, seq = _log_probs(S, theta, "S")
    return RichResult(payload={
        "estimate": float(logs.sum()),
        "per_token": [float(v) for v in logs], "n": len(seq),
        "method": "pseudo-log-likelihood (Kamath Eq 6.10)"})


def cheatsheet():
    return "km086: sum of log P(token | rest of the sentence)"


# compact alias per ledger/NAMING.md
kamathch6pll = kamath_ch6_pll
