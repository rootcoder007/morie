# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ViLBERT: dual-stream vision-language transformer."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsdp import geron_scaled_dot_product

__all__ = ["geron_vilbert"]


def _lcg_matrix(shape, seed, scale=0.5):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def geron_vilbert(image, text, d_model=8, seed=0):
    """
    ViLBERT: dual-stream vision-language transformer.

    Formula: two transformer streams (image, text) with co-attention layers

    The defining feature is **co-attention**, and it is implemented
    literally: each stream keeps its own representation, and in the
    co-attentional layer the *queries come from one modality while the
    keys and values come from the other*. Two attention passes run
    (delegated to :func:`morie.fn.hmsdp.geron_scaled_dot_product`):

    * image queries over text keys/values -> ``attention_v2t`` (regions x tokens);
    * text queries over image keys/values -> ``attention_t2v`` (tokens x regions).

    That is the difference from a single-stream model like VideoBERT,
    where one sequence holds both modalities: here the streams stay
    separate and exchange information only through the swapped queries,
    so each modality keeps its own depth and width.

    Parameters
    ----------
    image : array-like
        Region features (n_regions, d_v), or an (H, W) map that is
        flattened into H*W single-feature regions.
    text : array-like
        Token features (n_tokens, d_t), or a 1-D sequence of token ids
        which are embedded deterministically.
    d_model : int, default 8
        Shared co-attention width (>= 1).
    seed : int, default 0
        LCG seed for the projections and embeddings.

    Returns
    -------
    result : RichResult
        Keys: image_out, text_out, attention_v2t, attention_t2v, pooled,
        n_regions, n_tokens, estimate, n, method.

    Examples
    --------
    >>> r = geron_vilbert([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1], d_model=4)
    >>> int(r["n_regions"]), int(r["n_tokens"])
    (3, 2)
    >>> r["attention_v2t"].shape, r["attention_t2v"].shape
    ((3, 2), (2, 3))
    >>> [round(float(v), 12) for v in r["attention_v2t"].sum(axis=1)]
    [1.0, 1.0, 1.0]
    >>> r["image_out"].shape, r["text_out"].shape
    ((3, 4), (2, 4))
    >>> r["pooled"].shape
    (4,)

    References
    ----------
    Géron Ch 16
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 1:
        img = img.reshape(-1, 1)
    if img.ndim == 3:
        img = img.reshape(-1, img.shape[2])
    if img.ndim != 2 or img.size == 0:
        raise ValueError("geron_vilbert: image must be (n_regions, d_v) or an (H, W) map")
    if not np.all(np.isfinite(img)):
        raise ValueError("geron_vilbert: image contains non-finite values")
    d = int(d_model)
    if d < 1:
        raise ValueError(f"geron_vilbert: d_model must be >= 1, got {d}")

    txt = np.asarray(text)
    if txt.ndim == 1:
        if not np.all(np.equal(np.mod(txt.astype(float), 1), 0)):
            raise ValueError("geron_vilbert: a 1-D text input must contain integer token ids")
        tid = txt.astype(int)
        if np.any(tid < 0):
            raise ValueError("geron_vilbert: token ids must be non-negative")
        Emb = _lcg_matrix((int(tid.max()) + 1, d), int(seed) + 1)
        Tf = Emb[tid]
    else:
        Tf = np.asarray(txt, dtype=float)
        if Tf.ndim != 2 or Tf.size == 0:
            raise ValueError("geron_vilbert: text must be (n_tokens, d_t) features or a 1-D id sequence")
        if not np.all(np.isfinite(Tf)):
            raise ValueError("geron_vilbert: text contains non-finite values")
    if Tf.shape[0] == 0 or img.shape[0] == 0:
        raise ValueError("geron_vilbert: both streams must be non-empty for co-attention")

    Wv = _lcg_matrix((img.shape[1], d), int(seed) + 2)
    Wt = _lcg_matrix((Tf.shape[1], d), int(seed) + 3) if Tf.shape[1] != d else np.eye(d)
    Hv = img @ Wv
    Ht = Tf @ Wt

    Wq_v, Wk_t, Wv_t = (_lcg_matrix((d, d), int(seed) + 10 + i) for i in range(3))
    Wq_t, Wk_v, Wv_v = (_lcg_matrix((d, d), int(seed) + 20 + i) for i in range(3))

    v2t = geron_scaled_dot_product(Hv @ Wq_v, Ht @ Wk_t, Ht @ Wv_t, d_k=d)
    t2v = geron_scaled_dot_product(Ht @ Wq_t, Hv @ Wk_v, Hv @ Wv_v, d_k=d)

    img_out = Hv + np.asarray(v2t["Y"], dtype=float)
    txt_out = Ht + np.asarray(t2v["Y"], dtype=float)
    pooled = 0.5 * (img_out.mean(axis=0) + txt_out.mean(axis=0))

    return RichResult(
        title="ViLBERT co-attention",
        summary_lines=[
            ("Regions", int(img.shape[0])),
            ("Tokens", int(Tf.shape[0])),
            ("Co-attention width", d),
        ],
        interpretation=(
            "Swapping the queries is the whole mechanism: each stream asks questions of the other's "
            "content while keeping its own representation, so the modalities need not share depth or width."
        ),
        payload={
            "image_out": img_out,
            "text_out": txt_out,
            "attention_v2t": np.asarray(v2t["attention"], dtype=float),
            "attention_t2v": np.asarray(t2v["attention"], dtype=float),
            "pooled": pooled,
            "image_hidden": Hv,
            "text_hidden": Ht,
            "n_regions": int(img.shape[0]),
            "n_tokens": int(Tf.shape[0]),
            "estimate": float(np.max(v2t["attention"])),
            "n": int(img.shape[0] + Tf.shape[0]),
            "method": "Dual-stream co-attention: image queries over text keys/values and vice versa (hmsdp)",
        },
    )


def cheatsheet():
    return "hmvilb: ViLBERT: dual-stream vision-language transformer"


# compact alias per ledger/NAMING.md
geronvilbert = geron_vilbert
