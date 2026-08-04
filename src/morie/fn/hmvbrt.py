# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VideoBERT: transformer on discretized video tokens + text."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsdp import geron_scaled_dot_product
from .hmsftm import geron_softmax_function

__all__ = ["geron_videobert"]


def _lcg_matrix(shape, seed, scale=0.5):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def geron_videobert(video_tokens, text_tokens, d_model=8, mask_positions=None, mask_prob=0.25, seed=0):
    """
    VideoBERT: transformer on discretized video tokens + text.

    Formula: joint MLM on video+text tokens

    VideoBERT's contribution is representational, not architectural: video
    is vector-quantised into a *discrete* vocabulary so it can be poured
    into the same masked-language-model objective as text. That is what
    happens here -- the two token streams are embedded into one sequence
    with modality-typed embeddings and a ``[SEP]``-style boundary, one
    joint self-attention block runs over the whole sequence (delegated to
    :func:`morie.fn.hmsdp.geron_scaled_dot_product`), and the masked
    positions are scored with a softmax head over the joint vocabulary.

    Because attention is joint rather than per-stream, a text position can
    attend to video tokens directly; the fraction of attention mass that
    crosses the modality boundary is measured and returned, which is the
    only honest way to say whether the model is fusing anything.

    Parameters
    ----------
    video_tokens : array-like of int
        Quantised video token ids (non-empty).
    text_tokens : array-like of int
        Text token ids (non-empty). Ids share a joint vocabulary with the
        video stream after an offset is applied internally.
    d_model : int, default 8
        Embedding width (>= 1).
    mask_positions : sequence of int, optional
        Positions (into the concatenated sequence) to mask; by default
        every `1/mask_prob`-th position is chosen deterministically.
    mask_prob : float, default 0.25
        Masking rate used when `mask_positions` is not given, in (0, 1).
    seed : int, default 0
        LCG seed for the embeddings.

    Returns
    -------
    result : RichResult
        Keys: loss, attention, masked, predictions, cross_modal_mass,
        n_video, n_text, estimate, n, method.

    Examples
    --------
    >>> import numpy as np
    >>> r = geron_videobert([0, 1, 2, 1], [0, 1], d_model=4)
    >>> int(r["n_video"]), int(r["n_text"])
    (4, 2)
    >>> r["attention"].shape
    (6, 6)
    >>> [round(float(v), 12) for v in r["attention"].sum(axis=1)]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> bool(0.0 <= r["cross_modal_mass"] <= 1.0)
    True
    >>> bool(r["loss"] > 0)
    True
    >>> len(r["masked"]) >= 1
    True

    References
    ----------
    Géron Ch 16
    """
    v = np.asarray(video_tokens).ravel()
    t = np.asarray(text_tokens).ravel()
    for nm, a in (("video_tokens", v), ("text_tokens", t)):
        if a.size == 0:
            raise ValueError(f"geron_videobert: {nm} is empty; the joint objective needs both modalities")
        if not np.all(np.equal(np.mod(a.astype(float), 1), 0)):
            raise ValueError(f"geron_videobert: {nm} must contain integer token ids")
        if np.any(a.astype(int) < 0):
            raise ValueError(f"geron_videobert: {nm} ids must be non-negative")
    v = v.astype(int)
    t = t.astype(int)
    d = int(d_model)
    if d < 1:
        raise ValueError(f"geron_videobert: d_model must be >= 1, got {d}")

    n_v, n_t = v.size, t.size
    T = n_v + n_t
    v_vocab = int(v.max()) + 1
    ids = np.concatenate([v, t + v_vocab])  # joint vocabulary: text ids offset past the video codebook
    V = v_vocab + int(t.max()) + 1
    modality = np.concatenate([np.zeros(n_v, dtype=int), np.ones(n_t, dtype=int)])

    if mask_positions is None:
        p = float(mask_prob)
        if not (0.0 < p < 1.0):
            raise ValueError(f"geron_videobert: mask_prob must lie in (0, 1), got {p}")
        stride = max(1, int(round(1.0 / p)))
        masked = list(range(0, T, stride))
    else:
        masked = [int(m) for m in mask_positions]
        if not masked:
            raise ValueError("geron_videobert: mask_positions is empty; MLM needs at least one masked token")
        bad = [m for m in masked if not (0 <= m < T)]
        if bad:
            raise ValueError(f"geron_videobert: mask position(s) {bad} lie outside 0..{T - 1}")
    if len(masked) >= T:
        raise ValueError("geron_videobert: every position is masked; there is no context left to predict from")

    E = _lcg_matrix((V, d), int(seed) + 1)
    Mod = _lcg_matrix((2, d), int(seed) + 2, 0.2)
    H = E[ids] + Mod[modality]
    H[masked] = _lcg_matrix((1, d), int(seed) + 3, 0.1)  # [MASK] embedding replaces the token

    Wq, Wk, Wv = (_lcg_matrix((d, d), int(seed) + 10 + i) for i in range(3))
    att = geron_scaled_dot_product(H @ Wq, H @ Wk, H @ Wv, d_k=d)
    Y = np.asarray(att["Y"], dtype=float)
    A = np.asarray(att["attention"], dtype=float)

    Wout = _lcg_matrix((d, V), int(seed) + 20)
    losses = []
    preds = []
    for m in masked:
        p_dist = np.asarray(geron_softmax_function(Y[m] @ Wout)["p"], dtype=float)
        losses.append(-float(np.log(max(float(p_dist[ids[m]]), np.finfo(float).tiny))))
        preds.append(int(np.argmax(p_dist)))
    loss = float(np.mean(losses))

    cross = float(np.mean([A[i, modality != modality[i]].sum() for i in range(T)]))

    return RichResult(
        title="VideoBERT joint masked modelling",
        summary_lines=[
            ("Video tokens", n_v),
            ("Text tokens", n_t),
            ("Masked positions", len(masked)),
            ("Joint vocabulary", V),
            ("Cross-modal attention mass", cross),
            ("MLM loss", loss),
        ],
        interpretation=(
            "Quantising video into tokens is what makes the text objective reusable; if the cross-modal "
            "attention mass is near zero the two streams are being modelled side by side, not fused."
        ),
        payload={
            "loss": loss,
            "token_losses": np.asarray(losses, dtype=float),
            "attention": A,
            "hidden": Y,
            "masked": masked,
            "predictions": preds,
            "targets": ids[masked],
            "cross_modal_mass": cross,
            "n_video": int(n_v),
            "n_text": int(n_t),
            "vocab_size": int(V),
            "estimate": loss,
            "n": int(T),
            "method": "Joint MLM over concatenated video/text tokens with modality embeddings and shared self-attention",
        },
    )


def cheatsheet():
    return "hmvbrt: VideoBERT: transformer on discretized video tokens + text"


# compact alias per ledger/NAMING.md
geronvideobert = geron_videobert
