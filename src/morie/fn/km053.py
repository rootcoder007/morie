# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.12: the prefix-tuning objective."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_prefix_tuning_obj"]


def kamath_ch3_prefix_tuning_obj(phi, x, y, h, Y_idx=None):
    """max_phi log p_phi(y|x) = sum_{i in Y_idx} log p_phi(z_i | h_<i).

    A SUM (not a mean) of log probabilities over the target index set.
    ``phi`` is the caller's prefix-parameterised model, a callable
    (z_i, h_prefix) -> probability in (0, 1]; ``y`` is the target token
    sequence z; ``h`` the per-position activations, so ``h[:i]`` is the
    book's h_<i; ``x`` the source, carried for bookkeeping. ``Y_idx``
    defaults to every target position.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.12, printed
    p. 110.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch3_prefix_tuning_obj(
    ...     lambda z, hp: 0.5, "summarise:", ["a", "b"], [0.0, 1.0])
    >>> abs(out["estimate"] - 2 * math.log(0.5)) < 1e-12
    True
    """
    if not callable(phi):
        raise ValueError("phi must be a callable (z_i, h_prefix) -> "
                         "probability.")
    toks = list(y)
    states = list(h)
    if not toks:
        raise ValueError("the target sequence y is empty.")
    if len(states) != len(toks):
        raise ValueError(
            f"h has {len(states)} activations for {len(toks)} target "
            "tokens; h_<i must exist at every scored position.")
    idx = list(range(len(toks))) if Y_idx is None else [int(i) for i in Y_idx]
    if not idx:
        raise ValueError("Y_idx is empty; a sum over no positions is "
                         "undefined, not 0.")
    if len(set(idx)) != len(idx):
        raise ValueError("Y_idx contains duplicates.")
    if any(i < 0 or i >= len(toks) for i in idx):
        raise ValueError("an index in Y_idx lies outside the target.")
    logs = []
    for i in idx:
        p = float(phi(toks[i], states[:i]))
        if not (0.0 < p <= 1.0):
            raise ValueError(
                f"phi returned {p:.6g} at position {i}; it must lie in "
                "(0, 1].")
        logs.append(math.log(p))
    arr = np.asarray(logs, dtype=float)
    return RichResult(payload={
        "estimate": float(arr.sum()), "per_position": [float(v) for v in arr],
        "positions_scored": idx, "prompt": x, "n": len(toks),
        "method": "prefix-tuning objective (Kamath Eq 3.12)"})


def cheatsheet():
    return "km053: sum_{i in Y_idx} log p_phi(z_i | h_<i)"
