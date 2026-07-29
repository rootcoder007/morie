# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bahdanau (additive) attention over encoder states."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bahdanau_attention"]


def geron_bahdanau_attention(h, s_prev, W, U, v, b=None):
    """
    Bahdanau (additive) attention over encoder states.

    Formula: e_ij = v^T tanh(W h_i + U s_{j-1}); alpha_ij = softmax(e_ij)

    Parameters
    ----------
    h : array-like, shape (T, d_h)
        Encoder hidden states.
    s_prev : array-like, shape (d_s,)
        Previous decoder state.
    W : array-like, shape (d_a, d_h)
        Projection of the encoder states.
    U : array-like, shape (d_a, d_s)
        Projection of the decoder state.
    v : array-like, shape (d_a,)
        Alignment read-out vector.
    b : array-like, shape (d_a,), optional
        Bias inside the tanh.

    Returns
    -------
    result : RichResult
        Keys: alpha, scores, context, entropy, argmax, estimate, n, method.

    Examples
    --------
    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_bahdanau_attention([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], I, I, [1.0, 0.0])
    >>> [round(float(s), 6) for s in r["scores"]]
    [0.761594, 0.0]
    >>> [round(float(a), 6) for a in r["alpha"]]
    [0.6817, 0.3183]
    >>> [round(float(c), 6) for c in r["context"]]
    [0.6817, 0.3183]

    References
    ----------
    Géron Ch 14
    """
    H = np.asarray(h, dtype=float)
    if H.ndim == 1:
        H = H.reshape(1, -1)
    if H.ndim != 2:
        raise ValueError(f"geron_bahdanau_attention: h must be 2-D (T, d_h), got ndim={H.ndim}")
    T, d_h = H.shape
    if T == 0:
        raise ValueError("geron_bahdanau_attention: h has no time steps")
    s = np.asarray(s_prev, dtype=float).ravel()
    Wm = np.asarray(W, dtype=float)
    Um = np.asarray(U, dtype=float)
    vv = np.asarray(v, dtype=float).ravel()
    if Wm.ndim != 2 or Um.ndim != 2:
        raise ValueError("geron_bahdanau_attention: W and U must be 2-D matrices")
    if Wm.shape[1] != d_h:
        raise ValueError(f"geron_bahdanau_attention: W has {Wm.shape[1]} columns but h has {d_h} features")
    if Um.shape[1] != s.size:
        raise ValueError(f"geron_bahdanau_attention: U has {Um.shape[1]} columns but s_prev has {s.size} entries")
    if Wm.shape[0] != Um.shape[0]:
        raise ValueError(
            f"geron_bahdanau_attention: W and U must share the alignment dimension "
            f"(got {Wm.shape[0]} and {Um.shape[0]})"
        )
    d_a = Wm.shape[0]
    if vv.size != d_a:
        raise ValueError(f"geron_bahdanau_attention: v has {vv.size} entries but the alignment dim is {d_a}")
    bias = np.zeros(d_a) if b is None else np.asarray(b, dtype=float).ravel()
    if bias.size != d_a:
        raise ValueError(f"geron_bahdanau_attention: b has {bias.size} entries but the alignment dim is {d_a}")

    pre = H @ Wm.T + (Um @ s)[None, :] + bias[None, :]
    scores = np.tanh(pre) @ vv
    z = scores - float(np.max(scores))
    ez = np.exp(z)
    alpha = ez / float(np.sum(ez))
    context = alpha @ H
    nz = alpha > 0
    entropy = float(-np.sum(alpha[nz] * np.log(alpha[nz])))

    return RichResult(
        title="Bahdanau (additive) attention",
        summary_lines=[("Source length", T), ("Attention entropy", entropy), ("Argmax step", int(np.argmax(alpha)))],
        payload={
            "alpha": alpha,
            "scores": scores,
            "context": context,
            "entropy": entropy,
            "argmax": int(np.argmax(alpha)),
            "estimate": float(np.max(alpha)),
            "n": int(T),
            "method": "Bahdanau additive attention e = v^T tanh(W h + U s)",
        },
    )


def cheatsheet():
    return "hmbdn: Bahdanau (additive) attention over encoder states"
