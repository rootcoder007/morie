# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flamingo: visual dialogue with interleaved image and text."""

import numpy as np

from ._richresult import RichResult
from .grca import geron_cross_attention

__all__ = ["geron_flamingo"]


def geron_flamingo(images, text, latents=None, W_Q=None, W_K=None, W_V=None, gate=0.0, image_index=None):
    """
    Flamingo: visual dialogue with interleaved image and text.

    Formula: frozen LM + gated cross-attention to perceiver-encoded images

    Both mechanisms are computed, not described.

    **Perceiver resampler**: a fixed number of learned latents attend to
    the image features, so however many patch features an image produces,
    the language model always sees exactly ``len(latents)`` visual tokens.
    That is what makes a variable number of images affordable.

    **Gated cross-attention**: the visual contribution enters the frozen
    language model as ``h + tanh(gate) * xattn(h, visual)``. At
    initialisation ``gate = 0``, so ``tanh(0) = 0`` and the output is
    *exactly* the frozen LM's hidden states -- the new layers are the
    identity on day one, which is why adding them cannot damage the
    pretrained model. That identity is checked here, not asserted:
    ``is_identity_at_init`` compares the output with the input.

    Attention itself is DELEGATED to
    :func:`morie.fn.grca.geron_cross_attention` for both stages.

    ``image_index`` gives, per text position, which image that position
    may attend to (Flamingo's interleaving mask: text attends only to the
    most recent preceding image). Positions before any image get no
    visual input at all.

    Parameters
    ----------
    images : array-like, shape (n_features, d)
        Visual features of one image (e.g. ViT patch outputs).
    text : array-like, shape (T, d)
        Frozen LM hidden states.
    latents : array-like, shape (R, d), optional
        Perceiver latents; default one latent of ones.
    W_Q, W_K, W_V : array-like, optional
        Projections; default identity.
    gate : float, default 0.0
        Pre-tanh gate value.
    image_index : array-like of int, optional
        -1 marks a text position with no preceding image.

    Returns
    -------
    result : RichResult
        Keys: output, visual_tokens, cross_attention, gate_value,
        is_identity_at_init, n_visual_tokens, delta_norm, estimate, n,
        method.

    Examples
    --------
    At the initial gate the layer is exactly the identity on the frozen
    hidden states:

    >>> r = geron_flamingo([[1.0], [3.0]], [[2.0], [4.0]])
    >>> r["output"]
    [[2.0], [4.0]]
    >>> r["is_identity_at_init"]
    True
    >>> r["gate_value"]
    0.0

    The resampler compresses any number of image features to a fixed
    number of visual tokens -- here one latent averaging two features:

    >>> r["n_visual_tokens"]
    1
    >>> [round(v, 6) for v in r["visual_tokens"][0]]
    [2.0]

    Opening the gate lets the visual signal through, scaled by
    ``tanh(gate)``:

    >>> r2 = geron_flamingo([[1.0], [3.0]], [[0.0], [0.0]], gate=1.0)
    >>> [round(v, 6) for v in r2["output"][0]]
    [1.523188]
    >>> round(r2["delta_norm"], 6) > 0
    True

    A text position with no preceding image receives nothing:

    >>> r3 = geron_flamingo([[1.0], [3.0]], [[0.0], [0.0]], gate=1.0, image_index=[-1, 0])
    >>> r3["output"][0]
    [0.0]
    >>> round(r3["output"][1][0], 6)
    1.523188

    References
    ----------
    Géron Ch 16
    """
    V = np.atleast_2d(np.asarray(images, dtype=float))
    Hs = np.atleast_2d(np.asarray(text, dtype=float))
    if V.size == 0 or Hs.size == 0:
        raise ValueError("geron_flamingo: images and text must be non-empty")
    if V.shape[1] != Hs.shape[1]:
        raise ValueError(f"geron_flamingo: image width {V.shape[1]} != text width {Hs.shape[1]}")
    if not np.all(np.isfinite(V)) or not np.all(np.isfinite(Hs)):
        raise ValueError("geron_flamingo: images and text must be finite")
    d = Hs.shape[1]
    T = Hs.shape[0]

    L = np.ones((1, d)) if latents is None else np.atleast_2d(np.asarray(latents, dtype=float))
    if L.shape[1] != d:
        raise ValueError(f"geron_flamingo: latents width {L.shape[1]} != model width {d}")
    if L.shape[0] < 1:
        raise ValueError("geron_flamingo: at least one perceiver latent is required")

    I = np.eye(d)
    Wq = I if W_Q is None else np.atleast_2d(np.asarray(W_Q, dtype=float))
    Wk = I if W_K is None else np.atleast_2d(np.asarray(W_K, dtype=float))
    Wv = I if W_V is None else np.atleast_2d(np.asarray(W_V, dtype=float))
    g = float(gate)
    if not np.isfinite(g):
        raise ValueError(f"geron_flamingo: gate must be finite, got {gate!r}")

    # Stage 1: perceiver resampler -- latents attend to the image features.
    resample = geron_cross_attention(np.zeros_like(L) if latents is None else L, V, Wq, Wk, Wv)
    visual = np.atleast_2d(np.asarray(resample["output"], dtype=float))

    # Stage 2: gated cross-attention from text into the visual tokens.
    xattn = geron_cross_attention(Hs, visual, Wq, Wk, Wv)
    X = np.atleast_2d(np.asarray(xattn["output"], dtype=float))

    if image_index is not None:
        idx = np.asarray(image_index).ravel().astype(int)
        if idx.size != T:
            raise ValueError(f"geron_flamingo: image_index has {idx.size} entries but there are {T} text positions")
        if np.any(idx > 0):
            raise ValueError("geron_flamingo: this module handles one image; image_index entries must be 0 or -1")
        X = np.where(idx[:, None] < 0, 0.0, X)

    out = Hs + np.tanh(g) * X
    identity = bool(np.allclose(out, Hs))

    return RichResult(
        title="Flamingo gated cross-attention",
        summary_lines=[("Visual tokens", int(visual.shape[0])), ("tanh(gate)", float(np.tanh(g)))],
        interpretation="tanh(0) = 0, so a freshly initialised Flamingo layer is exactly the frozen language model.",
        payload={
            "output": out.tolist(),
            "visual_tokens": visual.tolist(),
            "cross_attention": X.tolist(),
            "attention_weights": xattn["attention_weights"],
            "resampler_weights": resample["attention_weights"],
            "gate_value": float(np.tanh(g)),
            "gate": g,
            "is_identity_at_init": identity,
            "n_visual_tokens": int(visual.shape[0]),
            "n_image_features": int(V.shape[0]),
            "delta_norm": float(np.linalg.norm(out - Hs)),
            "estimate": float(np.linalg.norm(out - Hs)),
            "n": int(T),
            "method": "perceiver resampler plus tanh-gated cross-attention; attention delegated to grca",
        },
    )


def cheatsheet():
    return "hmflmg: Flamingo: visual dialogue with interleaved image and text"
