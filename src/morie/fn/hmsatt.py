# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-attention: Q=K=V come from same input."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsdp import geron_scaled_dot_product

__all__ = ["geron_self_attention"]


def geron_self_attention(X, W_Q, W_K, W_V, mask=None):
    """
    Self-attention: Q=K=V come from same input.

    Formula: Att(X W_Q, X W_K, X W_V) for the same X

    Only the projections differ between self-attention and general
    attention, so the kernel is delegated to
    :func:`morie.fn.hmsdp.geron_scaled_dot_product` and this function is
    responsible for the three projections and the shape contract.

    Parameters
    ----------
    X : array-like
        Sequence, shape (T, d_model).
    W_Q, W_K : array-like
        Projections (d_model, d_k); both must have the same width.
    W_V : array-like
        Projection (d_model, d_v).
    mask : array-like, optional
        Passed straight through to the attention kernel; a lower-triangular
        (T, T) mask gives masked (causal) self-attention.

    Returns
    -------
    result : RichResult
        Keys: Y, attention, Q, K, V, estimate, n, method.

    Examples
    --------
    Identity projections on orthogonal rows: each token attends most to
    itself, and the attention rows are proper distributions.

    >>> X = [[1.0, 0.0], [0.0, 1.0]]
    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_self_attention(X, I, I, I)
    >>> r["Y"].shape
    (2, 2)
    >>> [round(float(v), 12) for v in r["attention"].sum(axis=1)]
    [1.0, 1.0]
    >>> bool(r["attention"][0, 0] > r["attention"][0, 1])
    True

    References
    ----------
    Géron Ch 15
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(1, -1)
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError("geron_self_attention: X must be a non-empty (T, d_model) matrix")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("geron_self_attention: X contains non-finite values")
    d_model = Xa.shape[1]
    mats = {}
    for nm, W in (("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        A = np.asarray(W, dtype=float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if A.ndim != 2:
            raise ValueError(f"geron_self_attention: {nm} must be a 2-D projection matrix")
        if A.shape[0] != d_model:
            raise ValueError(
                f"geron_self_attention: X has d_model={d_model} but {nm} has {A.shape[0]} rows"
            )
        if not np.all(np.isfinite(A)):
            raise ValueError(f"geron_self_attention: {nm} contains non-finite values")
        mats[nm] = A
    if mats["W_Q"].shape[1] != mats["W_K"].shape[1]:
        raise ValueError(
            f"geron_self_attention: W_Q projects to {mats['W_Q'].shape[1]} dims but W_K to "
            f"{mats['W_K'].shape[1]}; queries and keys must land in the same space"
        )

    Q = Xa @ mats["W_Q"]
    K = Xa @ mats["W_K"]
    V = Xa @ mats["W_V"]
    inner = geron_scaled_dot_product(Q, K, V, d_k=Q.shape[1], mask=mask)

    return RichResult(
        title="Self-attention",
        summary_lines=[
            ("Tokens", int(Xa.shape[0])),
            ("d_model", int(d_model)),
            ("d_k", int(Q.shape[1])),
            ("d_v", int(V.shape[1])),
        ],
        interpretation=(
            "Every token is compared with every other token in the same sequence, so the receptive "
            "field is the whole sequence in one layer (cost O(T^2))."
        ),
        payload={
            "Y": np.asarray(inner["Y"], dtype=float),
            "attention": np.asarray(inner["attention"], dtype=float),
            "scores": np.asarray(inner["scores"], dtype=float),
            "Q": Q,
            "K": K,
            "V": V,
            "estimate": float(np.max(inner["attention"])),
            "n": int(Xa.shape[0]),
            "method": "Self-attention: shared-source projections into hmsdp",
        },
    )


def cheatsheet():
    return "hmsatt: Self-attention: Q=K=V come from same input"
