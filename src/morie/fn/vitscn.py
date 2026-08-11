# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scaled cosine attention (Swin Transformer V2).

Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao,
Y., Zhang, Z., Dong, L., Wei, F. and Guo, B. (2022), "Swin Transformer
V2: Scaling Up Capacity and Resolution", CVPR 2022, arXiv:2111.09883,
Section 3.2 ("Scaling Up Model Capacity"):

    Sim(q_i, k_j) = cos(q_i, k_j) / tau + B_ij

replacing the dot product; tau is a learnable per-head, per-layer
scalar constrained to be larger than 0.01, and B_ij is the relative
position bias. The cosine is naturally normalised, which prevents the
attention maps of large models from being dominated by a few pixel
pairs.

MISATTRIBUTED LEAD, RECORDED: the stub cited "Liu et al (2021), ViT-2
log-scaled attention". There is no "ViT-2"; scaled cosine attention is
Swin Transformer V2 (the arXiv preprint is Nov 2021, the paper CVPR
2022). Implemented from the real source.

Source: fetched-wave3/liu-etal-2022-swin-v2-arxiv2111.09883.pdf
(Section 3.2).
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vitscn", "vit_scaled_cosine"]


def vitscn(q, k, v, tau=0.1, B=None):
    """Scaled cosine attention (Liu et al. 2022, arXiv:2111.09883, Sec 3.2).

    Parameters
    ----------
    q : array-like, shape (n_q, d)
    k : array-like, shape (n_k, d)
    v : array-like, shape (n_k, d_v)
    tau : float
        Learnable temperature; the paper constrains tau > 0.01.
    B : array-like, shape (n_q, n_k), optional
        Relative position bias added AFTER the scaled cosine; 0 if
        omitted.

    Returns
    -------
    result : RichResult
        Keys: output (n_q x d_v), weights, similarities (pre-softmax
        cos/tau + B), tau, estimate, n, method.
    """
    Qa = np.atleast_2d(np.asarray(q, dtype=float))
    Ka = np.atleast_2d(np.asarray(k, dtype=float))
    Va = np.atleast_2d(np.asarray(v, dtype=float))
    t = float(tau)
    if not t > 0.01:
        raise ValueError(
            f"vitscn: tau must exceed 0.01 (paper constraint), got {t}")
    if Qa.shape[1] != Ka.shape[1]:
        raise ValueError(f"vitscn: q width {Qa.shape[1]} != k width {Ka.shape[1]}")
    if Ka.shape[0] != Va.shape[0]:
        raise ValueError(
            f"vitscn: k has {Ka.shape[0]} rows but v has {Va.shape[0]}")
    for name, arr in (("q", Qa), ("k", Ka), ("v", Va)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"vitscn: {name} contains non-finite values")
    nq, nk = Qa.shape[0], Ka.shape[0]
    if B is None:
        Bm = [[0.0] * nk for _ in range(nq)]
    else:
        Ba = np.atleast_2d(np.asarray(B, dtype=float))
        if Ba.shape != (nq, nk):
            raise ValueError(
                f"vitscn: B must be ({nq}, {nk}), got {Ba.shape}")
        Bm = [[float(x) for x in row] for row in Ba]
    qn = [math.sqrt(sum(float(x) * float(x) for x in row)) for row in Qa]
    kn = [math.sqrt(sum(float(x) * float(x) for x in row)) for row in Ka]
    if any(x == 0.0 for x in qn) or any(x == 0.0 for x in kn):
        raise ValueError("vitscn: cosine similarity undefined for a zero row")
    S = []
    for i in range(nq):
        row = []
        for j in range(nk):
            dot = sum(float(a) * float(b) for a, b in zip(Qa[i], Ka[j]))
            row.append(dot / (qn[i] * kn[j]) / t + Bm[i][j])
        S.append(row)
    W = []
    for row in S:
        m = max(row)
        e = [math.exp(x - m) for x in row]
        z = sum(e)
        W.append([x / z for x in e])
    out = np.asarray(W, dtype=float) @ Va
    return RichResult(payload={
        "output": [[float(x) for x in row] for row in out],
        "weights": W,
        "similarities": S,
        "tau": t,
        "estimate": float(out[0][0] if out.ndim == 2 else out[0]),
        "n": int(nq),
        "method": "scaled cosine attention cos(q,k)/tau + B (Liu et al. 2022, Sec 3.2)",
    })


vit_scaled_cosine = vitscn


def cheatsheet():
    return "vitscn: scaled cosine attention (Swin V2, Liu et al. 2022, arXiv:2111.09883, Sec 3.2)"
