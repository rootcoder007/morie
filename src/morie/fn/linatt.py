# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linformer linear-complexity attention via low-rank projection.

Wang, S., Li, B. Z., Khabsa, M., Fang, H. and Ma, H. (2020),
"Linformer: Self-Attention with Linear Complexity", arXiv:2006.04768,
Eq 7: with projection matrices E, F in R^(k x n) applied to the KEY
and VALUE layers,

    head = softmax( Q (E K)^T / sqrt(d_k) ) . (F V)

so the attention matrix P-bar is n x k instead of n x n and the cost
drops from O(n^2) to O(n k). Here Q, K, V are the already-projected
query/key/value matrices (the paper's Q W_Q etc.), and E, F are the
shared linear projections of Theorem 2.

With k = n and E = F = I_n this reduces EXACTLY to standard scaled
dot-product attention -- that limiting case is the test anchor.

Source: fetched-wave3/wang-etal-2020-linformer-arxiv2006.04768.pdf
(Section 4, Eq 7).
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["linatt", "linformer_linear_attention"]


def _softmax_rows(S):
    out = []
    for row in S:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        t = sum(e)
        out.append([v / t for v in e])
    return out


def linatt(Q, K, V, E, F):
    """Linformer attention (Wang et al. 2020, arXiv:2006.04768, Eq 7).

    Parameters
    ----------
    Q : array-like, shape (n, d_k)
    K : array-like, shape (n, d_k)
    V : array-like, shape (n, d_v)
    E : array-like, shape (k, n)
        Key projection.
    F : array-like, shape (k, n)
        Value projection.

    Returns
    -------
    result : RichResult
        Keys: output (n x d_v), weights (the n x k attention matrix),
        projected_K (k x d_k), projected_V (k x d_v), k, estimate, n,
        method.
    """
    Qa = np.atleast_2d(np.asarray(Q, dtype=float))
    Ka = np.atleast_2d(np.asarray(K, dtype=float))
    Va = np.atleast_2d(np.asarray(V, dtype=float))
    Ea = np.atleast_2d(np.asarray(E, dtype=float))
    Fa = np.atleast_2d(np.asarray(F, dtype=float))
    n, dk = Qa.shape
    if Ka.shape[1] != dk:
        raise ValueError(f"linatt: K width {Ka.shape[1]} != Q width {dk}")
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(
            f"linatt: K has {Ka.shape[0]} rows but V has {Va.shape[0]}")
    if Ea.shape[1] != Ka.shape[0]:
        raise ValueError(
            f"linatt: E must be (k, n) with n = {Ka.shape[0]}, got {Ea.shape}")
    if Fa.shape[1] != Va.shape[0]:
        raise ValueError(
            f"linatt: F must be (k, n) with n = {Va.shape[0]}, got {Fa.shape}")
    if Ea.shape[0] != Fa.shape[0]:
        raise ValueError(
            f"linatt: E and F must share k, got {Ea.shape[0]} and {Fa.shape[0]}")
    for name, arr in (("Q", Qa), ("K", Ka), ("V", Va), ("E", Ea), ("F", Fa)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"linatt: {name} contains non-finite values")
    EK = Ea @ Ka                       # (k, d_k)
    FV = Fa @ Va                       # (k, d_v)
    S = (Qa @ EK.T) * (1.0 / math.sqrt(dk))   # (n, k)
    P = _softmax_rows([[float(v) for v in row] for row in S])
    Pm = np.asarray(P, dtype=float)
    out = Pm @ FV                      # (n, d_v)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in out],
        "weights": P,
        "projected_K": [[float(v) for v in row] for row in EK],
        "projected_V": [[float(v) for v in row] for row in FV],
        "k": int(Ea.shape[0]),
        "estimate": float(out[0][0] if out.ndim == 2 else out[0]),
        "n": int(n),
        "method": "Linformer low-rank attention (Wang et al. 2020, Eq 7)",
    })


linformer_linear_attention = linatt


def cheatsheet():
    return "linatt: Linformer low-rank attention (Wang et al. 2020, arXiv:2006.04768, Eq 7)"
