# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.21: entropy-based attention regularisation (EAR)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_ear_entropy_reg"]


def kamath_ch6_ear_entropy_reg(A, L=None, lam=1.0):
    """R = -lam sum_{l=1..L} entropy(A)_l.

    MINIMISING R maximises attention entropy, spreading each layer's
    attention over the context instead of letting it lock onto a few
    stereotypical tokens -- hence the leading minus sign, which makes R
    non-positive for non-negative lam. ``A`` is one attention matrix
    per layer, each row a distribution over context positions; a
    layer's entropy is the mean over its rows, in nats.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.21, printed
    p. 244.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch6_ear_entropy_reg([[[0.5, 0.5]]], lam=1.0)
    >>> abs(out["estimate"] + math.log(2.0)) < 1e-12
    True
    >>> kamath_ch6_ear_entropy_reg([[[1.0, 0.0]]])["estimate"]
    -0.0
    """
    layers = list(A)
    if not layers:
        raise ValueError("A is empty; there are no layers to regularise.")
    if L is not None and int(L) != len(layers):
        raise ValueError(
            f"L = {int(L)} contradicts the {len(layers)} attention "
            "matrices supplied.")
    lam = float(lam)
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and non-negative.")
    ents = []
    for i, M in enumerate(layers):
        Am = np.atleast_2d(np.asarray(M, dtype=float))
        if Am.size == 0:
            raise ValueError(f"layer {i}'s attention matrix is empty.")
        if np.any(Am < 0):
            raise ValueError(f"layer {i} has a negative attention weight.")
        sums = Am.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > 1e-8):
            raise ValueError(
                f"every attention row of layer {i} must sum to 1; got "
                f"{[float(v) for v in sums[:4]]!r}.")
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(Am > 0, -Am * np.log(Am), 0.0)
        ents.append(float(terms.sum(axis=1).mean()))
    total = float(sum(ents))
    return RichResult(payload={
        "estimate": -lam * total, "per_layer_entropy": ents,
        "total_entropy": total, "lam": lam, "n": len(layers),
        "method": "entropy-based attention regularisation "
                  "(Kamath Eq 6.21)"})


def cheatsheet():
    return "km097: -lam * summed per-layer mean attention entropy"
