# morie.fn -- function file (rootcoder007/morie)
"""DINO self-distillation cross-entropy."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['dinoloss', 'dino_self_distill']


def dinoloss(s_logits, t_logits, tau_s=0.1, tau_t=0.04, C=None):
    """DINO self-distillation cross-entropy.

    The teacher is a past average of the student, so this is distillation with no teacher given in advance. The loss is a cross-entropy, not a symmetric divergence: the gradient flows only through the student. Softmaxes are computed by subtracting the row maximum first, which changes nothing mathematically and prevents an overflow that would otherwise appear only for confident logits.


    Formula: P_s = softmax(s/tau_s), P_t = softmax((t - C)/tau_t), loss = -(1/B) sum_i sum_k P_t(i,k) log P_s(i,k)

    Parameters
    ----------
    s_logits : array-like, shape (B, K)
        Student logits.
    t_logits : array-like, shape (B, K)
        Teacher logits.
    tau_s : float
        Student temperature.
    tau_t : float
        Teacher temperature.
    C : array-like, optional
        Teacher centre; zeros if omitted.

    Returns
    -------
    RichResult
        ``loss``, ``per_view``, ``p_s``, ``p_t``, ``B``, ``K``.

    References
    ----------
    Caron, Touvron, Misra, Jegou, Mairal, Bojanowski and Joulin (2021),
    Emerging Properties in Self-Supervised Vision Transformers,
    ICCV/arXiv:2104.14294.  Verified against the paper: equation (1)
    for the temperature softmax, equation (4) for the centre update,
    and Algorithm 1's pseudocode for the order of centre-then-sharpen.
    """
    Sm = C.mat(s_logits); Tm = C.mat(t_logits)
    B = len(Sm); K = len(Sm[0])
    if len(Tm) != B or len(Tm[0]) != K:
        raise ValueError("student and teacher logits must have the same shape")
    c = [0.0] * K if C is None else C.vec(C)
    if len(c) != K:
        raise ValueError("C must have length K")

    def sm(row, tau, off):
        z = [(row[j] - off[j]) / float(tau) for j in range(K)]
        mx = max(z)
        e = [math.exp(v - mx) for v in z]
        s = sum(e)
        return [v / s for v in e]

    zero = [0.0] * K
    Ps = [sm(r, tau_s, zero) for r in Sm]
    Pt = [sm(r, tau_t, c) for r in Tm]
    per = [-sum(Pt[i][j] * math.log(Ps[i][j]) for j in range(K)) for i in range(B)]
    return RichResult(payload={
        "loss": sum(per) / B, "per_view": per, "p_s": Ps, "p_t": Pt,
        "B": B, "K": K, "method": "DINO self-distillation loss"})


dino_self_distill = dinoloss


def cheatsheet():
    return "dinopr: DINO self-distillation cross-entropy."
