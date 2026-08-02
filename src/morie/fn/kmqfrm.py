# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q-Former: learnable query tokens cross-attend to visual features
(BLIP-2)."""

from . import _array_core as np

from ._richresult import RichResult
from .attsdp import scaled_dot_product_attention

__all__ = ["kamath_q_former"]


def kamath_q_former(queries, visual_features, W_out=None):
    """Z = CrossAttn(Q = learnable queries, K = V = visual features).

    Cross-attention IS scaled dot-product attention with K and V drawn
    from the other modality, so the computation is DELEGATED to
    ``morie.fn.attsdp`` instead of duplicated. What Q-Former adds is
    the fixed-size bottleneck: N query tokens summarise however many
    patches the vision encoder produced, and that compression ratio is
    reported.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Q-Former (BLIP-2).

    Examples
    --------
    >>> out = kamath_q_former([[1.0, 0.0]],
    ...     [[1.0, 0.0], [0.0, 1.0]])
    >>> round(out["Z"][0][0], 6)
    0.669762
    >>> out["n_queries"], out["n_patches"]
    (1, 2)
    """
    Q = np.atleast_2d(np.asarray(queries, dtype=float))
    F = np.atleast_2d(np.asarray(visual_features, dtype=float))
    if Q.shape[0] == 0:
        raise ValueError("a Q-Former with no query tokens outputs nothing.")
    if F.shape[0] == 0:
        raise ValueError("no visual features to attend to.")
    if Q.shape[1] != F.shape[1]:
        raise ValueError(
            f"queries are {Q.shape[1]}-dim but the visual features are "
            f"{F.shape[1]}-dim; cross-attention needs a shared width "
            "(project the visual side first).")
    att = scaled_dot_product_attention(Q, F, F)
    Z = np.asarray(att["output"], dtype=float)
    if W_out is not None:
        W = np.atleast_2d(np.asarray(W_out, dtype=float))
        if W.shape[0] != Z.shape[1]:
            raise ValueError(
                f"W_out expects {W.shape[0]}-dim inputs but Z is "
                f"{Z.shape[1]}-dim.")
        Z = Z @ W
    return RichResult(payload={
        "Z": [[float(v) for v in row] for row in Z],
        "attention": att["attention"],
        "n_queries": int(Q.shape[0]),
        "n_patches": int(F.shape[0]),
        "compression": float(F.shape[0]) / Q.shape[0],
        "estimate": float(Z[0, 0]),
        "n": int(Q.shape[0]),
        "method": "Q-Former cross-attention (delegates to attsdp)"})


def cheatsheet():
    return "kmqfrm: attsdp with Q = learned queries, K = V = patches"
