# morie.fn -- function file (rootcoder007/morie)
"""Chunked causal attention for long-context efficiency."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_chunked_attention"]


def causal_chunked_attention(Q, K, V, chunk_size, n_chunks_back=None):
    r"""Blockwise causal attention over chunks of the sequence.

    Splits the sequence into chunks of size c and lets a query attend
    to its own chunk (causally) plus whole earlier chunks. Restricting
    to ``n_chunks_back`` previous chunks turns the :math:`O(L^2)` cost
    into :math:`O(L c (n+1))` -- the standard long-context trade-off;
    with ``n_chunks_back=None`` every earlier chunk is visible and the
    output equals full causal attention exactly.

    Parameters
    ----------
    Q, K, V : array-like, shape (L, d) -- V may be (L, dv)
        Query, key and value matrices.
    chunk_size : int
        Chunk length c.
    n_chunks_back : int, optional
        How many complete earlier chunks stay visible. None = all.

    Returns
    -------
    RichResult
        keys: ``output`` (L, dv), ``attention`` (L, L) with zeros where
        masked, ``mask`` (L, L) boolean, ``chunk_size``,
        ``n_chunks_back``, ``density`` (fraction of attended pairs),
        ``method``.

    References
    ----------
    Kamath, U., Graham, K. L. & Emara, W. (2022). *Transformers for
    Machine Learning: A Deep Dive*. Chapman & Hall/CRC. Ch. 7
    (efficient / sparse attention: blockwise and local patterns).
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be 2-D.")
    L, d = Q.shape
    if K.shape != (L, d) or V.shape[0] != L:
        raise ValueError("Q, K must share shape and V must have L rows.")
    c = int(chunk_size)
    if c < 1:
        raise ValueError(f"chunk_size must be at least 1, got {c}.")
    if n_chunks_back is not None and int(n_chunks_back) < 0:
        raise ValueError("n_chunks_back must be nonnegative.")

    idx = np.arange(L)
    ci = idx // c
    causal = idx[None, :] <= idx[:, None]
    same = ci[None, :] == ci[:, None]
    if n_chunks_back is None:
        window = ci[None, :] <= ci[:, None]
    else:
        back = int(n_chunks_back)
        window = (ci[None, :] <= ci[:, None]) & (ci[None, :] >= ci[:, None] - back)
    mask = causal & (same | window)

    scores = Q @ K.T / np.sqrt(d)
    scores = np.where(mask, scores, -np.inf)
    m = scores.max(axis=1, keepdims=True)
    ex = np.where(mask, np.exp(scores - m), 0.0)
    attn = ex / ex.sum(axis=1, keepdims=True)

    return RichResult(
        payload={
            "output": attn @ V,
            "attention": attn,
            "mask": mask,
            "chunk_size": c,
            "n_chunks_back": n_chunks_back,
            "density": float(mask.sum() / (L * L)),
            "method": "Chunked causal attention (own chunk + n earlier chunks)",
        }
    )


def cheatsheet():
    return "nchunk: causal attention restricted to own chunk + n earlier chunks"
