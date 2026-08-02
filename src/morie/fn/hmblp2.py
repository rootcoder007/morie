# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BLIP-2: frozen image encoder + lightweight Q-Former."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_blip2"]


def _lcg(size, seed, scale=0.1):
    s = int(seed) % 2**32
    out = np.empty(int(size))
    for i in range(int(size)):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out


def _init(shape, seed, scale=0.1):
    return _lcg(int(np.prod(shape)), seed, scale).reshape(shape)


def _softmax(z, axis=-1):
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def geron_blip2(image, text, n_query=4, d_query=8, d_llm=None, temperature=1.0, seed=0):
    """
    BLIP-2: frozen image encoder + lightweight Q-Former.

    Formula: Q-Former bridges frozen ViT and frozen LLM

    The image and text arrive as embeddings from encoders that stay frozen --
    nothing here updates them. The only trainable module is the Q-Former: a
    fixed set of learned query vectors that cross-attend the image patch
    features and are then projected into the LLM's input width. That is the
    whole point of BLIP-2, and the returned parameter counts make the
    asymmetry explicit.

    Parameters
    ----------
    image : array-like, shape (n_patches, d_v)
        Frozen image-encoder patch features (a 1-D vector is treated as one
        patch).
    text : array-like, shape (d_t,) or (n_tokens, d_t)
        Frozen text embeddings; token matrices are mean-pooled.
    n_query : int
        Number of learned queries (>= 1).
    d_query : int
        Q-Former working width (>= 1).
    d_llm : int, optional
        LLM input width; defaults to `d_query`.
    temperature : float
        Positive temperature for the image-text similarity.
    seed : int
        LCG seed for the Q-Former parameters.

    Returns
    -------
    result : RichResult
        Keys: query_output, llm_input, attention, similarity, trainable_params,
        estimate, n, method.

    Examples
    --------
    >>> img = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> r = geron_blip2(img, [1.0, 0.0], n_query=4, d_query=8, d_llm=6)
    >>> r["llm_input"].shape
    (4, 6)
    >>> r["attention"].shape
    (4, 3)

    Cross-attention rows are distributions over the image patches:

    >>> [round(float(v), 12) for v in r["attention"].sum(axis=1)]
    [1.0, 1.0, 1.0, 1.0]

    The pipeline is deterministic, and only the Q-Former is trainable:

    >>> bool(np.allclose(geron_blip2(img, [1.0, 0.0])["llm_input"], geron_blip2(img, [1.0, 0.0])["llm_input"]))
    True
    >>> r["frozen_encoders"]
    ('image', 'text')

    References
    ----------
    Géron Ch 16
    """
    V = np.asarray(image, dtype=float)
    if V.ndim == 1:
        V = V.reshape(1, -1)
    if V.ndim != 2:
        raise ValueError(f"geron_blip2: image must be 1-D or 2-D patch features, got ndim={V.ndim}")
    if V.shape[0] == 0:
        raise ValueError("geron_blip2: image has no patches")
    if not np.all(np.isfinite(V)):
        raise ValueError("geron_blip2: image features must be finite")
    Tt = np.asarray(text, dtype=float)
    if Tt.ndim == 1:
        Tt = Tt.reshape(1, -1)
    if Tt.ndim != 2 or Tt.shape[0] == 0:
        raise ValueError("geron_blip2: text must be a 1-D embedding or a 2-D token matrix")
    if not np.all(np.isfinite(Tt)):
        raise ValueError("geron_blip2: text features must be finite")
    Q = int(n_query)
    dq = int(d_query)
    if Q < 1 or dq < 1:
        raise ValueError("geron_blip2: n_query and d_query must both be >= 1")
    dl = dq if d_llm is None else int(d_llm)
    if dl < 1:
        raise ValueError("geron_blip2: d_llm must be >= 1")
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"geron_blip2: temperature must be positive, got {tau}")

    n_patches, dv = V.shape
    dt = Tt.shape[1]
    queries = _init((Q, dq), seed + 11)
    Wk = _init((dv, dq), seed + 22)
    Wv = _init((dv, dq), seed + 33)
    Wo = _init((dq, dq), seed + 44)
    Wllm = _init((dq, dl), seed + 55)
    Wtxt = _init((dt, dq), seed + 66)

    K = V @ Wk
    Vv = V @ Wv
    attn = _softmax(queries @ K.T / np.sqrt(dq), axis=1)
    qout = (attn @ Vv) @ Wo
    llm_input = qout @ Wllm

    text_vec = Tt.mean(axis=0) @ Wtxt
    qn = qout / np.clip(np.linalg.norm(qout, axis=1, keepdims=True), 1e-12, None)
    tn = text_vec / max(float(np.linalg.norm(text_vec)), 1e-12)
    sims = qn @ tn
    # BLIP-2 keeps the strongest query as the image-text score.
    similarity = float(np.max(sims) / tau)

    trainable = int(Q * dq + dv * dq * 2 + dq * dq + dq * dl + dt * dq)

    return RichResult(
        title="BLIP-2 Q-Former",
        summary_lines=[("Queries", Q), ("Patches", n_patches), ("Image-text similarity", similarity)],
        payload={
            "query_output": qout,
            "llm_input": llm_input,
            "attention": attn,
            "similarity": similarity,
            "query_similarities": sims,
            "trainable_params": trainable,
            "frozen_encoders": ("image", "text"),
            "d_llm": dl,
            "estimate": similarity,
            "n": int(n_patches),
            "method": "BLIP-2: learned queries cross-attending frozen image features, projected to the LLM width",
        },
    )


def cheatsheet():
    return "hmblp2: BLIP-2: frozen image encoder + lightweight Q-Former"
