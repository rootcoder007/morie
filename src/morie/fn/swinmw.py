# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Swin window multi-head self-attention with relative position bias.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and
Guo, B. (2021), "Swin Transformer: Hierarchical Vision Transformer
using Shifted Windows", ICCV 2021, arXiv:2103.14030, Eq 4:

    Attention(Q, K, V) = SoftMax( Q K^T / sqrt(d) + B ) V

computed INSIDE each non-overlapping M x M window, where the bias
B in R^(M^2 x M^2) is looked up from a smaller parameterised table
B-hat in R^((2M-1) x (2M-1)) by the relative position of the two
tokens along each axis (each lies in [-M+1, M-1]).

The sibling modules grswin/hmswin implement windowed attention
WITHOUT the relative position bias, so this is not aliased to them;
the bias lookup (the part Eq 4 adds) is implemented here and the
result reduces exactly to theirs when the table is zero.

Source: fetched-wave3/liu-etal-2021-swin-transformer-arxiv2103.14030.pdf
(Section 3.2, Eq 4 and the paragraph "Relative position bias").
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["swinmw", "swin_msa_window"]


def _window_bias(M, table):
    # B[p, q] = table[di + M - 1][dj + M - 1] with p = (i1, j1),
    # q = (i2, j2) flattened row-major and (di, dj) = (i1-i2, j1-j2).
    n = M * M
    B = [[0.0] * n for _ in range(n)]
    for p in range(n):
        i1, j1 = divmod(p, M)
        for q in range(n):
            i2, j2 = divmod(q, M)
            B[p][q] = float(table[i1 - i2 + M - 1][j1 - j2 + M - 1])
    return B


def swinmw(x, window_size, relative_bias=None, WQ=None, WK=None, WV=None):
    """Swin window MSA with relative position bias (Liu et al. 2021, Eq 4).

    Parameters
    ----------
    x : array-like, shape (H, W, d)
        Feature map; window_size must divide H and W.
    window_size : int
        Window side M.
    relative_bias : array-like, shape (2M-1, 2M-1), optional
        The parameterised table B-hat; zeros if omitted.
    WQ, WK, WV : array-like, shape (d, d_k) / (d, d_k) / (d, d_v), optional
        Projections; identity if omitted.

    Returns
    -------
    result : RichResult
        Keys: output (H x W x d_v), bias (M^2 x M^2 matrix B),
        n_windows, tokens_per_window, estimate, n, method.
    """
    # the native array core has no 3-D container, so the (H, W, d) map
    # is parsed as nested lists directly
    try:
        A = [[[float(v) for v in cell] for cell in row] for row in x]
    except TypeError:
        raise ValueError("swinmw: x must be a (H, W, d) nested array")
    H = len(A)
    W = len(A[0]) if H else 0
    d = len(A[0][0]) if W else 0
    if H == 0 or W == 0 or d == 0:
        raise ValueError("swinmw: x must be a non-empty (H, W, d) array")
    for row in A:
        if len(row) != W or any(len(cell) != d for cell in row):
            raise ValueError("swinmw: x is ragged")
        for cell in row:
            if not all(math.isfinite(v) for v in cell):
                raise ValueError("swinmw: x contains non-finite values")
    M = int(window_size)
    if M < 1 or H % M or W % M:
        raise ValueError(
            f"swinmw: window_size must divide H and W, got {M} for ({H}, {W})")
    eye = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    Wq = np.atleast_2d(np.asarray(WQ if WQ is not None else eye, dtype=float))
    Wk = np.atleast_2d(np.asarray(WK if WK is not None else eye, dtype=float))
    Wv = np.atleast_2d(np.asarray(WV if WV is not None else eye, dtype=float))
    if Wq.shape[0] != d or Wk.shape[0] != d or Wv.shape[0] != d:
        raise ValueError("swinmw: projection rows must equal d")
    if Wq.shape[1] != Wk.shape[1]:
        raise ValueError("swinmw: WQ and WK widths must match")
    dk = Wq.shape[1]
    dv = Wv.shape[1]
    if relative_bias is None:
        table = [[0.0] * (2 * M - 1) for _ in range(2 * M - 1)]
    else:
        T = np.atleast_2d(np.asarray(relative_bias, dtype=float))
        if T.shape != (2 * M - 1, 2 * M - 1):
            raise ValueError(
                f"swinmw: relative_bias must be ({2 * M - 1}, {2 * M - 1}), got {T.shape}")
        table = [[float(v) for v in row] for row in T]
    B = _window_bias(M, table)
    out = [[[0.0] * dv for _ in range(W)] for _ in range(H)]
    scale = 1.0 / math.sqrt(dk)
    n_windows = 0
    for h0 in range(0, H, M):
        for w0 in range(0, W, M):
            n_windows += 1
            toks = [A[h0 + i][w0 + j] for i in range(M) for j in range(M)]
            X = np.asarray(toks, dtype=float)
            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv
            S = (Q @ K.T) * scale
            Sm = [[float(S[p][q]) + B[p][q] for q in range(M * M)]
                  for p in range(M * M)]
            Wt = []
            for row in Sm:
                mx = max(row)
                e = [math.exp(v - mx) for v in row]
                z = sum(e)
                Wt.append([v / z for v in e])
            O = np.asarray(Wt, dtype=float) @ V
            for p in range(M * M):
                i, j = divmod(p, M)
                out[h0 + i][w0 + j] = [float(v) for v in O[p]]
    return RichResult(payload={
        "output": out,
        "bias": B,
        "n_windows": n_windows,
        "tokens_per_window": M * M,
        "estimate": float(out[0][0][0]),
        "n": int(H * W),
        "method": "Swin window MSA softmax(QK^T/sqrt(d) + B)V (Liu et al. 2021, Eq 4)",
    })


swin_msa_window = swinmw


def cheatsheet():
    return "swinmw: Swin window MSA with relative position bias (Liu et al. 2021, arXiv:2103.14030, Eq 4)"
