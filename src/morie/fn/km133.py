# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.5: the CLIP image-to-text contrastive loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_clip_image_to_text"]


def kamath_ch9_clip_image_to_text(V, L, sigma, N=None):
    r"""L_i2t = -(1/N) sum_i log[ exp(V_i.L_i/s) / sum_j exp(V_i.L_j/s) ].

    Rows of ``V`` and ``L`` are the (already normalized) image and
    text embeddings of a batch of N pairs; ``sigma`` is the
    temperature. The log-sum-exp is computed with the max subtracted,
    so a small temperature does not overflow.

    Eq 9.6 is this same expression with the modalities swapped, so
    ``morie.fn.km134`` delegates here rather than duplicating it.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.5, printed
    p. 386.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_clip_image_to_text([[1.0, 0.0], [0.0, 1.0]],
    ...                                     [[1.0, 0.0], [0.0, 1.0]], 1.0)
    >>> abs(out["estimate"] - (math.log(math.e + 1) - 1)) < 1e-12
    True
    """
    Vm = np.atleast_2d(np.asarray(V, dtype=float))
    Lm = np.atleast_2d(np.asarray(L, dtype=float))
    s = float(sigma)
    if s <= 0:
        raise ValueError(f"the temperature must be positive; got {s}.")
    if Vm.shape[0] != Lm.shape[0]:
        raise ValueError(
            f"the batch sizes differ: {Vm.shape[0]} image embeddings "
            f"vs {Lm.shape[0]} text embeddings.")
    if Vm.shape[1] != Lm.shape[1]:
        raise ValueError(
            f"embedding widths differ: {Vm.shape[1]} vs {Lm.shape[1]}.")
    if Vm.shape[0] == 0:
        raise ValueError("the batch is empty.")
    if N is not None and int(N) != Vm.shape[0]:
        raise ValueError(
            f"N = {N} contradicts the batch size {Vm.shape[0]}.")
    logits = Vm @ Lm.T / s
    m = logits.max(axis=1, keepdims=True)
    lse = (m.ravel() + np.log(np.exp(logits - m).sum(axis=1)))
    per = lse - np.diag(logits)
    return RichResult(payload={
        "estimate": float(per.mean()),
        "per_pair": [float(v) for v in per],
        "logits": [[float(v) for v in row] for row in logits],
        "temperature": s, "n": int(Vm.shape[0]),
        "method": "CLIP image-to-text contrastive loss (Kamath Eq 9.5)"})


def cheatsheet():
    return "km133: row-wise InfoNCE over image-text logits at temp sigma"
