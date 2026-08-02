# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perceiver IO: cross-attention encoder plus a query-driven decoder."""

from . import _array_core as np

from ._richresult import RichResult
from .hmprcv import _softmax_rows, geron_perceiver

__all__ = ["geron_perceiver_io"]


def geron_perceiver_io(x, latents, queries, n_iter=2, W_q=None, W_k=None, W_v=None):
    """
    Perceiver IO: adds a cross-attention output decoder.

    Formula: Perceiver + decoder cross-attention from output queries

    The encoder is DELEGATED to :func:`~morie.fn.hmprcv.geron_perceiver`.
    What Perceiver IO adds is the other end: instead of pooling the
    latents into one vector, a set of output QUERIES cross-attends into
    them, so the output shape is set by the queries and not by the
    architecture. One query gives a classification; one query per pixel
    gives a dense prediction; the encoder is unchanged either way.

    The decoder costs O(M*L) for M queries, so the whole model stays
    linear in both the input and the output size -- the property the
    original Perceiver lacked.

    Parameters
    ----------
    x : array-like, shape (N, D)
    latents : array-like, shape (L, D)
    queries : array-like, shape (M, D)
        Output queries; their count fixes the output count.
    n_iter : int, default 2
    W_q, W_k, W_v : array-like, optional
        Encoder projections, as in the Perceiver.

    Returns
    -------
    result : RichResult
        Keys: outputs, decoder_attention, latents, encoder_attention,
        decoder_cost, estimate, n, method.

    Examples
    --------
    Three queries against a two-latent encoder give three outputs, each
    a convex combination of the latents:

    >>> x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    >>> r = geron_perceiver_io(x, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> r["outputs"].shape, r["decoder_attention"].shape
    ((3, 2), (3, 2))
    >>> [round(float(v), 12) for v in r["decoder_attention"].sum(axis=1)]
    [1.0, 1.0, 1.0]

    One query gives one output, with the same encoder:

    >>> geron_perceiver_io(x, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]])["outputs"].shape
    (1, 2)

    References
    ----------
    Geron Ch 16
    """
    enc = geron_perceiver(x, latents, n_iter=n_iter, W_q=W_q, W_k=W_k, W_v=W_v)
    Z = np.asarray(enc["latents"], dtype=float)
    Q = np.atleast_2d(np.asarray(queries, dtype=float))
    if Q.ndim != 2 or Q.size == 0:
        raise ValueError(f"geron_perceiver_io: queries must be a non-empty (M, D) array, got shape {Q.shape}")
    if not np.all(np.isfinite(Q)):
        raise ValueError("geron_perceiver_io: queries contain non-finite values")
    if Q.shape[1] != Z.shape[1]:
        raise ValueError(
            f"geron_perceiver_io: queries are {Q.shape[1]}-dimensional but the latents are {Z.shape[1]}-dimensional"
        )

    dk = Z.shape[1]
    attn = _softmax_rows(Q @ Z.T / np.sqrt(dk))
    out = attn @ Z
    return RichResult(
        title="Perceiver IO",
        summary_lines=[("Queries", int(Q.shape[0])), ("Latents", int(Z.shape[0])), ("Output shape", out.shape)],
        interpretation="Output size is set by the queries, so encoder and decoder both stay linear in their inputs.",
        payload={
            "outputs": out,
            "decoder_attention": attn,
            "latents": Z,
            "encoder_attention": enc["attention"],
            "decoder_cost": int(Q.shape[0] * Z.shape[0]),
            "encoder_cost": int(enc["attention_cost"]),
            "estimate": out,
            "n": int(Q.shape[0]),
            "method": "Perceiver IO: encoder delegated to morie.fn.hmprcv, query cross-attention decoder",
        },
    )


def cheatsheet():
    return "hmprio: Perceiver IO with a query-driven output decoder"
