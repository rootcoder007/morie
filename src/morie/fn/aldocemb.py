# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mean-pooled document embedding with an attention mask
(Alammar Ch 8)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_document_embedding_pool"]


def alammar_document_embedding_pool(token_embeddings, attention_mask=None):
    """d = sum(mask_i h_i) / sum(mask_i): padding must NOT dilute the
    mean, which is the whole reason the mask is part of the formula.

    References: Alammar and Grootendorst, Ch 8 (sentence embeddings).

    Examples
    --------
    >>> alammar_document_embedding_pool([[2.0], [4.0], [99.0]],
    ...     [1, 1, 0])["embedding"]
    [3.0]
    """
    H = np.atleast_2d(np.asarray(token_embeddings, dtype=float))
    if attention_mask is None:
        m = np.ones(H.shape[0])
    else:
        m = np.atleast_1d(np.asarray(attention_mask, dtype=float))
    if len(m) != H.shape[0]:
        raise ValueError(
            f"mask length {len(m)} does not match {H.shape[0]} tokens.")
    if m.sum() == 0:
        raise ValueError("the mask excludes every token; an all-padding "
                         "document has no embedding.")
    d = (H * m[:, None]).sum(axis=0) / m.sum()
    return RichResult(payload={
        "embedding": [float(v) for v in d],
        "tokens_pooled": int(m.sum()),
        "estimate": float(d[0]), "n": H.shape[0],
        "method": "Masked mean pooling (Alammar Ch 8)"})


def cheatsheet():
    return "aldocemb: masked mean pool; padding never dilutes the mean"
