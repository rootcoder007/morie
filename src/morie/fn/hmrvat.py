# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RNN visual attention over a spatial feature map."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_rnn_visual_attention"]


def geron_rnn_visual_attention(features, h, W, U, v):
    """
    RNNs with visual attention over a spatial feature map.

    Formula: context_t = sum_{i,j} alpha_{ij,t} * feature_{i,j}

    The decoder's state asks a question of every spatial location at
    once: e_ij = v^T tanh(W f_ij + U h), softmaxed into weights that sum
    to one and used to average the feature vectors. The context is
    therefore always INSIDE the convex hull of the features -- attention
    can only select and blend, never invent -- and the weight map is the
    part you can show a person, which is why captioning models are
    inspectable in a way their unattended predecessors were not.

    Because the weights are recomputed from the current state, the same
    image yields a different context at every decoding step.

    Parameters
    ----------
    features : array-like, shape (H, W_, D) or (N, D)
        Spatial feature map; the grid is flattened internally.
    h : array-like, shape (d_h,)
        Decoder state.
    W : array-like, shape (k, D)
    U : array-like, shape (k, d_h)
    v : array-like, shape (k,)

    Returns
    -------
    result : RichResult
        Keys: context, alpha, alpha_map, scores, entropy, estimate, n,
        method.

    Examples
    --------
    Two orthogonal features and a zero state: only the first survives
    the v projection, so it gets the larger weight.

    >>> f = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_rnn_visual_attention(f, [0.0], np.eye(2), np.zeros((2, 1)), [1.0, 0.0])
    >>> round(float(r["alpha"].sum()), 12)
    1.0
    >>> bool(r["alpha"][0] > r["alpha"][1])
    True

    The context is a convex combination, so it never leaves the hull:

    >>> bool(0.0 <= r["context"][0] <= 1.0 and 0.0 <= r["context"][1] <= 1.0)
    True

    A spatial map keeps its shape in ``alpha_map``:

    >>> g = np.arange(12.0).reshape(2, 2, 3)
    >>> geron_rnn_visual_attention(g, [0.0], np.eye(3), np.zeros((3, 1)),
    ...                            [1.0, 0.0, 0.0])["alpha_map"].shape
    (2, 2)

    References
    ----------
    Geron Ch 16
    """
    F = np.asarray(features, dtype=float)
    if F.ndim == 3:
        grid = F.shape[:2]
        Ff = F.reshape(-1, F.shape[2])
    elif F.ndim == 2:
        grid = None
        Ff = F
    else:
        raise ValueError(f"geron_rnn_visual_attention: features must be (H, W, D) or (N, D), got ndim={F.ndim}")
    if Ff.size == 0:
        raise ValueError("geron_rnn_visual_attention: features is empty")
    hv = np.atleast_1d(np.asarray(h, dtype=float)).ravel()
    Wm = np.atleast_2d(np.asarray(W, dtype=float))
    Um = np.atleast_2d(np.asarray(U, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    N, D = Ff.shape
    k = Wm.shape[0]
    if Wm.shape != (k, D):
        raise ValueError(f"geron_rnn_visual_attention: W has shape {Wm.shape}, expected (k, {D})")
    if Um.shape != (k, hv.size):
        raise ValueError(f"geron_rnn_visual_attention: U has shape {Um.shape}, expected ({k}, {hv.size})")
    if vv.size != k:
        raise ValueError(f"geron_rnn_visual_attention: v has {vv.size} entries but W has {k} rows")
    for name, arr in (("features", Ff), ("h", hv), ("W", Wm), ("U", Um), ("v", vv)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_rnn_visual_attention: {name} contains non-finite values")

    pre = Ff @ Wm.T + (Um @ hv)[None, :]
    scores = np.tanh(pre) @ vv
    e = np.exp(scores - scores.max())
    alpha = e / e.sum()
    context = alpha @ Ff
    nz = alpha[alpha > 0]
    entropy = float(-np.sum(nz * np.log(nz)))

    return RichResult(
        title="Visual attention context",
        summary_lines=[("Locations", int(N)), ("Entropy (nats)", entropy), ("Max weight", float(alpha.max()))],
        interpretation="The context is a convex blend of the features; the weight map is what you can show a person.",
        payload={
            "context": context,
            "alpha": alpha,
            "alpha_map": alpha.reshape(grid) if grid is not None else alpha,
            "scores": scores,
            "entropy": entropy,
            "estimate": context,
            "n": int(N),
            "method": "Additive (Bahdanau-style) attention over a spatial feature map",
        },
    )


def cheatsheet():
    return "hmrvat: RNN visual attention context over a feature map"
