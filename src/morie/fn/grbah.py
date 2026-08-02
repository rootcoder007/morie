# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bahdanau additive attention: score, softmax, context."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bahdanau_attention"]

_METHOD = "Bahdanau (additive) attention"


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def geron_bahdanau_attention(decoder_state, encoder_states, Wh, Ws, v):
    r"""Additive ("concat") attention over the encoder outputs.

    .. math::
        e_{ti} = v^{\top}\tanh(W_h h_t + W_s s_i), \qquad
        \alpha_{ti} = \operatorname{softmax}_i(e_{ti}), \qquad
        c_t = \sum_i \alpha_{ti} s_i

    Additive attention scores with a small feed-forward net rather than a
    dot product, so the decoder and encoder states are free to have
    different widths -- and unlike the dot product it needs no
    :math:`\sqrt{d_k}` rescaling, because the tanh already bounds the
    pre-projection.

    Parameters
    ----------
    decoder_state : array-like, shape (dh,)
        Current decoder hidden state :math:`h_t`.
    encoder_states : array-like, shape (T, ds)
        Encoder outputs :math:`s_1 \dots s_T`.
    Wh : array-like, shape (a, dh)
    Ws : array-like, shape (a, ds)
        Projections into the ``a``-dimensional alignment space.
    v : array-like, shape (a,)
        Alignment vector.

    Returns
    -------
    RichResult
        Payload keys ``context``, ``weights``, ``scores``, ``entropy``
        (of the attention distribution, in nats), ``argmax``,
        ``estimate`` (max attention weight), ``n``, ``method``.

    References
    ----------
    Géron Ch 14, Bahdanau Attention section.

    Examples
    --------
    With ``Ws = 0`` every encoder state scores identically, so attention
    is uniform and the context is the plain mean:

    >>> r = geron_bahdanau_attention([0.0], [[1.0], [3.0]], Wh=[[1.0]],
    ...                              Ws=[[0.0]], v=[1.0])
    >>> [round(w, 6) for w in r["weights"]]
    [0.5, 0.5]
    >>> [round(c, 6) for c in r["context"]]
    [2.0]

    Weights always form a distribution:

    >>> r2 = geron_bahdanau_attention([1.0], [[1.0], [3.0]], Wh=[[1.0]],
    ...                               Ws=[[2.0]], v=[1.0])
    >>> round(sum(r2["weights"]), 12)
    1.0
    >>> r2["argmax"]
    1
    """
    h = np.asarray(decoder_state, dtype=float).ravel()
    S = np.atleast_2d(np.asarray(encoder_states, dtype=float))
    Wh = np.atleast_2d(np.asarray(Wh, dtype=float))
    Ws = np.atleast_2d(np.asarray(Ws, dtype=float))
    v = np.asarray(v, dtype=float).ravel()
    if S.shape[0] == 0:
        raise ValueError("encoder_states is empty.")
    if Wh.shape[1] != h.size:
        raise ValueError(
            f"Wh has {Wh.shape[1]} columns but decoder_state has {h.size} entries."
        )
    if Ws.shape[1] != S.shape[1]:
        raise ValueError(
            f"Ws has {Ws.shape[1]} columns but encoder states have width {S.shape[1]}."
        )
    if Wh.shape[0] != Ws.shape[0]:
        raise ValueError(
            f"Wh and Ws must project into the same alignment space, got "
            f"{Wh.shape[0]} and {Ws.shape[0]} rows."
        )
    if v.size != Wh.shape[0]:
        raise ValueError(
            f"v has {v.size} entries but the alignment space is {Wh.shape[0]}-dimensional."
        )
    for name, arr in (("decoder_state", h), ("encoder_states", S), ("Wh", Wh), ("Ws", Ws), ("v", v)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    proj_h = Wh @ h                       # (a,)
    proj_s = S @ Ws.T                     # (T, a)
    scores = np.tanh(proj_h[None, :] + proj_s) @ v   # (T,)
    alpha = _softmax(scores)
    context = alpha @ S

    nz = alpha[alpha > 0]
    entropy = float(-np.sum(nz * np.log(nz)))

    return RichResult(
        title="Bahdanau attention",
        summary_lines=[("Source length", int(S.shape[0])), ("Entropy (nats)", entropy)],
        payload={
            "context": context.tolist(),
            "weights": alpha.tolist(),
            "scores": scores.tolist(),
            "entropy": entropy,
            "argmax": int(np.argmax(alpha)),
            "estimate": float(alpha.max()),
            "n": int(S.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbah: additive attention e=v^T tanh(Wh h + Ws s); softmax; context = sum alpha_i s_i"
