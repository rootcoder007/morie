# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Perceiver: cross-attention from learned latents to a large input."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_perceiver"]


def _softmax_rows(S):
    E = np.exp(S - S.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


def geron_perceiver(x, latents, n_iter=2, W_q=None, W_k=None, W_v=None):
    """
    Perceiver: cross-attention to learned latents from high-dimensional inputs.

    Formula: latents Q cross-attend to input K,V; iterate to refine

    Self-attention over N inputs costs O(N^2), which rules out raw
    pixels or audio samples. The Perceiver attends ONCE from a small set
    of L learned latents into the input, so the cost is O(L*N) with L
    fixed by the architecture rather than by the data. The saving is
    reported as ``attention_cost`` against ``self_attention_cost``.

    Iterating the cross-attention lets the latents re-query the input
    after they have been updated -- a second look with a better question.
    The projections default to the identity, so the plain form is
    scaled dot-product attention from latents to inputs.

    Parameters
    ----------
    x : array-like, shape (N, D)
        Input array (already flattened over space or time).
    latents : array-like, shape (L, D_lat)
        Learned latent array; ``D_lat`` must equal ``D`` unless
        projections are supplied.
    n_iter : int, default 2
        Cross-attention rounds.
    W_q : array-like, shape (D_lat, d), optional
    W_k, W_v : array-like, shape (D, d), optional

    Returns
    -------
    result : RichResult
        Keys: latents, attention, attention_cost, self_attention_cost,
        estimate, n, method.

    Examples
    --------
    Two orthogonal inputs and one latent aligned with the first: the
    attention weights are a distribution, and the latent moves toward
    what it attended to.

    >>> r = geron_perceiver([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], n_iter=1)
    >>> r["attention"].shape
    (1, 2)
    >>> round(float(r["attention"].sum()), 12)
    1.0
    >>> bool(r["attention"][0, 0] > r["attention"][0, 1])
    True

    The cost is linear in N, not quadratic:

    >>> int(r["attention_cost"]), int(r["self_attention_cost"])
    (2, 4)

    References
    ----------
    Geron Ch 16
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    L = np.atleast_2d(np.asarray(latents, dtype=float)).astype(float).copy()
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"geron_perceiver: x must be a non-empty (N, D) array, got shape {X.shape}")
    if L.ndim != 2 or L.size == 0:
        raise ValueError(f"geron_perceiver: latents must be a non-empty (L, D) array, got shape {L.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(L)):
        raise ValueError("geron_perceiver: x or latents contain non-finite values")
    T = int(n_iter)
    if T < 1:
        raise ValueError(f"geron_perceiver: n_iter must be >= 1, got {n_iter!r}")

    N, D = X.shape
    nl, Dl = L.shape
    Wq = np.eye(Dl) if W_q is None else np.atleast_2d(np.asarray(W_q, dtype=float))
    Wk = np.eye(D) if W_k is None else np.atleast_2d(np.asarray(W_k, dtype=float))
    Wv = np.eye(D) if W_v is None else np.atleast_2d(np.asarray(W_v, dtype=float))
    if Wq.shape[0] != Dl:
        raise ValueError(f"geron_perceiver: W_q has {Wq.shape[0]} rows but the latents are {Dl}-dimensional")
    if Wk.shape[0] != D or Wv.shape[0] != D:
        raise ValueError(f"geron_perceiver: W_k and W_v must have {D} rows to match x")
    if Wq.shape[1] != Wk.shape[1]:
        raise ValueError(f"geron_perceiver: query width {Wq.shape[1]} does not match key width {Wk.shape[1]}")
    if Wv.shape[1] != Dl:
        raise ValueError(
            f"geron_perceiver: values are {Wv.shape[1]}-dimensional but the latents are {Dl}-dimensional; "
            "the residual update needs them equal"
        )

    dk = Wq.shape[1]
    K = X @ Wk
    V = X @ Wv
    attn = None
    for _ in range(T):
        Q = L @ Wq
        attn = _softmax_rows(Q @ K.T / np.sqrt(dk))
        L = L + attn @ V

    return RichResult(
        title="Perceiver cross-attention",
        summary_lines=[("Inputs", int(N)), ("Latents", int(nl)), ("Rounds", T)],
        interpretation="Cost is O(L*N) with L fixed by the architecture, which is what makes raw inputs affordable.",
        payload={
            "latents": L,
            "attention": attn,
            "attention_cost": int(nl * N),
            "self_attention_cost": int(N * N),
            "n_iter": T,
            "estimate": L,
            "n": int(N),
            "method": "Perceiver: iterated latent-to-input scaled dot-product cross-attention",
        },
    )


def cheatsheet():
    return "hmprcv: Perceiver latent cross-attention over large inputs"
