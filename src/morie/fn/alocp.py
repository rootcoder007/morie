# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLIP-style symmetric contrastive loss (Radford et al. 2021;
Alammar Ch 9)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_openclip_contrastive"]


def alammar_openclip_contrastive(I_emb, T_emb, tau=0.07):
    """L2-normalise both towers; L = (CE(rows) + CE(columns)) / 2 on
    sim/tau with the diagonal as the target.

    References: Alammar and Grootendorst, Ch 9; Radford et al. (2021).
    """
    t = float(tau)
    if t <= 0:
        raise ValueError("the temperature must be positive.")
    I = np.atleast_2d(np.asarray(I_emb, dtype=float))
    T = np.atleast_2d(np.asarray(T_emb, dtype=float))
    if I.shape != T.shape:
        raise ValueError("image and text batches must align.")
    if I.shape[0] < 2:
        raise ValueError("need a batch of at least 2.")
    ni = np.linalg.norm(I, axis=1, keepdims=True)
    nt = np.linalg.norm(T, axis=1, keepdims=True)
    if np.any(ni == 0) or np.any(nt == 0):
        raise ValueError("zero embedding vectors have no direction.")
    I = I / ni; T = T / nt
    S = I @ T.T / t

    def ce_diag(M):
        Z = M - M.max(axis=1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        return -np.diag(logp)

    li = ce_diag(S)
    lt = ce_diag(S.T)
    loss = float((li.mean() + lt.mean()) / 2)
    return RichResult(payload={
        "estimate": loss, "image_to_text_loss": float(li.mean()),
        "text_to_image_loss": float(lt.mean()),
        "similarity_matrix": [[float(v * t) for v in r] for r in S],
        "n": I.shape[0],
        "method": "CLIP symmetric contrastive loss (Radford et al. 2021)"})


def cheatsheet():
    return "alocp: symmetric CE over the normalised similarity matrix"
