# morie.fn -- function file (rootcoder007/morie)
r"""Multi-head attention (scaled dot-product).

Self-attention mechanism with multiple representation subspaces.

References
----------
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
Attention is all you need.
In NIPS (pp. 5998-6008).
"""

__all__ = ["mhatt"]

from . import _array_core as np
from ._sci_core import softmax

from ._richresult import RichResult


def mhatt(
    query,
    key,
    value,
    num_heads=8,
    d_model=512,
):
    """
    Multi-head scaled dot-product attention (Vaswani et al. 2017, Sec. 3.2).

    The model dimension is split into ``num_heads`` subspaces of width
    ``d_k = d_model / num_heads``; each head attends on its own slice,

        head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i,

    and the heads are concatenated. No learned projections are passed
    in, so W_i^Q = W_i^K = W_i^V = W^O = I: head i sees columns
    i*d_k .. (i+1)*d_k - 1.

    Parameters
    ----------
    query : ndarray
        Query, shape (seq_q, d_model).
    key : ndarray
        Key, shape (seq_k, d_model).
    value : ndarray
        Value, shape (seq_k, d_model).
    num_heads : int, optional
        Number of attention heads. Default 8.
    d_model : int, optional
        Model dimension. Default 512.

    Returns
    -------
    dict
        Keys: 'output' (seq_q, d_model), 'attention_weights'
        (num_heads, seq_q, seq_k).

    Examples
    --------
    Two heads of width 1. Head 0 sees column 0, where the query matches
    key 0 exactly and key 1 not at all; head 1 sees column 1:

    >>> r = mhatt([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]],
    ...           [[10.0, 1.0], [20.0, 2.0]], num_heads=2, d_model=2)
    >>> w = r["attention_weights"].tolist()
    >>> [round(v, 6) for v in w[0][0]], [round(v, 6) for v in w[1][0]]
    ([0.731059, 0.268941], [0.5, 0.5])
    """
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)

    if d_model % num_heads != 0:
        raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
    for name, m in (("query", query), ("key", key), ("value", value)):
        if m.ndim != 2 or m.shape[1] != d_model:
            raise ValueError(f"{name} must be (seq_len, {d_model}), got shape {m.shape}")
    if key.shape[0] != value.shape[0]:
        raise ValueError("key and value must have the same sequence length")

    d_k = d_model // num_heads
    heads, weights = [], []
    for h in range(num_heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        scores = np.dot(query[:, sl], key[:, sl].T) / np.sqrt(d_k)
        w = softmax(scores, axis=-1)
        weights.append(np.asarray(w).tolist())
        heads.append(np.dot(w, value[:, sl]))
    output = np.concatenate(heads, axis=1)
    attention_weights = np.asarray(weights)

    return RichResult(payload={"output": output, "attention_weights": attention_weights,
                               "d_k": d_k, "num_heads": num_heads})


def cheatsheet() -> str:
    return "mhatt: mhatt(query, key, value, num_heads, d_model) -> Multi-head scaled dot-product attention."
