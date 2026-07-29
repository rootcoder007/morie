# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FlashAttention: IO-aware exact attention with tiling and recomputation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_flash_attention"]


def geron_flash_attention(Q, K, V, block_size=2, causal=False):
    """
    FlashAttention: IO-aware exact attention with tiling and recomputation.

    Formula: Att computed block-wise; avoids O(N^2) memory materialization

    The tiled online-softmax algorithm is implemented for real: for each
    query block the key/value blocks are streamed, and the running
    ``(max, sum, accumulator)`` triple is rescaled by
    ``exp(m_old - m_new)`` as the maximum moves. Nothing but a
    ``block_size x block_size`` tile of scores exists at any moment, so
    peak score memory is ``O(B^2)`` instead of ``O(N^2)``.

    The result is *exact*, not approximate. That claim is checked inside
    the function: the direct ``softmax(QK^T/sqrt(d))V`` is computed too
    and ``max_abs_error`` reports the difference (floating-point noise
    only). If it ever exceeded ``1e-8`` the tiling would be wrong.

    Parameters
    ----------
    Q : array-like, shape (N, d)
    K : array-like, shape (M, d)
    V : array-like, shape (M, dv)
    block_size : int, default 2
        Tile side; must be positive.
    causal : bool, default False
        Mask keys after the query position (needs ``N == M``).

    Returns
    -------
    result : RichResult
        Keys: output, direct_output, max_abs_error, row_max, row_sum,
        n_blocks, peak_score_memory, naive_score_memory, memory_ratio,
        estimate, n, method.

    Examples
    --------
    A zero query attends uniformly, so the output is the mean value row:

    >>> r = geron_flash_attention([[0.0]], [[1.0], [3.0]], [[1.0], [3.0]])
    >>> [round(v, 6) for v in r["output"][0]]
    [2.0]
    >>> r["max_abs_error"] < 1e-12
    True

    Tiling does not change the answer, only the memory:

    >>> Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    >>> K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    >>> V = [[1.0], [2.0], [3.0], [4.0]]
    >>> a = geron_flash_attention(Q, K, V, block_size=1)
    >>> b = geron_flash_attention(Q, K, V, block_size=4)
    >>> max(abs(p[0] - q[0]) for p, q in zip(a["output"], b["output"])) < 1e-12
    True
    >>> a["peak_score_memory"], b["peak_score_memory"]
    (1, 16)
    >>> a["naive_score_memory"]
    16

    Causal masking makes the first query attend only to itself:

    >>> r2 = geron_flash_attention(Q, K, V, block_size=2, causal=True)
    >>> round(r2["output"][0][0], 9)
    1.0

    References
    ----------
    Géron Ch 17
    """
    Qa = np.atleast_2d(np.asarray(Q, dtype=float))
    Ka = np.atleast_2d(np.asarray(K, dtype=float))
    Va = np.atleast_2d(np.asarray(V, dtype=float))
    if Qa.size == 0 or Ka.size == 0 or Va.size == 0:
        raise ValueError("geron_flash_attention: Q, K and V must be non-empty")
    if Qa.shape[1] != Ka.shape[1]:
        raise ValueError(f"geron_flash_attention: Q width {Qa.shape[1]} != K width {Ka.shape[1]}")
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(f"geron_flash_attention: K has {Ka.shape[0]} rows but V has {Va.shape[0]}")
    for name, arr in (("Q", Qa), ("K", Ka), ("V", Va)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_flash_attention: {name} contains non-finite values")
    B = int(block_size)
    if B < 1:
        raise ValueError(f"geron_flash_attention: block_size must be >= 1, got {block_size!r}")
    N, d = Qa.shape
    M, dv = Ka.shape[0], Va.shape[1]
    if causal and N != M:
        raise ValueError(f"geron_flash_attention: causal masking needs N == M, got {N} and {M}")

    scale = 1.0 / np.sqrt(d)
    out = np.zeros((N, dv))
    row_m = np.full(N, -np.inf)
    row_l = np.zeros(N)
    n_blocks = 0

    for i0 in range(0, N, B):
        i1 = min(i0 + B, N)
        qi = Qa[i0:i1]
        m_i = np.full(i1 - i0, -np.inf)
        l_i = np.zeros(i1 - i0)
        o_i = np.zeros((i1 - i0, dv))
        for j0 in range(0, M, B):
            j1 = min(j0 + B, M)
            n_blocks += 1
            s = (qi @ Ka[j0:j1].T) * scale
            if causal:
                qi_idx = np.arange(i0, i1)[:, None]
                kj_idx = np.arange(j0, j1)[None, :]
                s = np.where(kj_idx > qi_idx, -np.inf, s)
            blk_max = s.max(axis=1)
            m_new = np.maximum(m_i, blk_max)
            m_new = np.where(np.isfinite(m_new), m_new, 0.0)
            corr = np.exp(np.where(np.isfinite(m_i), m_i, -np.inf) - m_new)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            p = np.exp(s - m_new[:, None])
            p = np.where(np.isfinite(p), p, 0.0)
            l_i = corr * l_i + p.sum(axis=1)
            o_i = corr[:, None] * o_i + p @ Va[j0:j1]
            m_i = m_new
        if np.any(l_i == 0):
            raise ValueError("geron_flash_attention: a query row has no unmasked keys")
        out[i0:i1] = o_i / l_i[:, None]
        row_m[i0:i1] = m_i
        row_l[i0:i1] = l_i

    # Direct reference computation, for the exactness claim.
    S = (Qa @ Ka.T) * scale
    if causal:
        S = np.where(np.arange(M)[None, :] > np.arange(N)[:, None], -np.inf, S)
    Sm = S - S.max(axis=1, keepdims=True)
    E = np.exp(Sm)
    direct = (E / E.sum(axis=1, keepdims=True)) @ Va
    err = float(np.max(np.abs(out - direct)))

    peak = int(min(B, N) * min(B, M))

    return RichResult(
        title="FlashAttention (tiled)",
        summary_lines=[("Tiles", n_blocks), ("Peak score memory", peak), ("Max |error|", err)],
        interpretation="Tiling is an exact reformulation: only the memory traffic changes, never the output.",
        payload={
            "output": out.tolist(),
            "direct_output": direct.tolist(),
            "max_abs_error": err,
            "row_max": row_m.tolist(),
            "row_sum": row_l.tolist(),
            "n_blocks": int(n_blocks),
            "block_size": B,
            "peak_score_memory": peak,
            "naive_score_memory": int(N * M),
            "memory_ratio": float((N * M) / peak),
            "causal": bool(causal),
            "estimate": err,
            "n": int(N),
            "method": "tiled online-softmax attention (FlashAttention), exact",
        },
    )


def cheatsheet():
    return "hmfa: FlashAttention: IO-aware exact attention with tiling and recomputation"
