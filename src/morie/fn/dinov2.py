# morie.fn -- function file (rootcoder007/morie)
"""DINOv2 objective: DINO + iBOT + KoLeo."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .clipsi import l2_normalize
from .dinmlt import dino_softmax

__all__ = ["dino_v2_repr"]


def dino_v2_repr(x, student, teacher, tau=0.1, tau_t=0.04, mask=None,
                 w_ibot=1.0, w_koleo=0.1):
    """
    DINOv2 representation objective

    Formula: DINO + iBOT mask + KoLeo

    Three terms.  The image-level DINO cross-entropy between the
    sharpened teacher and the student; the patch-level iBOT
    cross-entropy over the MASKED patches only; and the KoLeo
    regulariser -mean log d_i, where d_i is the distance from
    embedding i to its nearest neighbour, which spreads the batch out
    and stops features from piling up.

    Parameters
    ----------
    x : array-like
        n x d matrix of L2-normalisable batch embeddings, for KoLeo.
    student : array-like
        (1 + P) x k logits: the class token followed by P patch tokens.
    teacher : array-like
        (1 + P) x k teacher logits, same layout.
    tau : float
        Student temperature.
    tau_t : float
        Teacher temperature.
    mask : array-like or None
        0/1 flag per patch; None masks every second patch.
    w_ibot, w_koleo : float
        Weights of the iBOT and KoLeo terms.

    Returns
    -------
    result : dict
        Keys: estimate (total loss), loss, dino, ibot, koleo, n_masked,
        n, d.

    References
    ----------
    Oquab et al. (2024), DINOv2: Learning Robust Visual Features
    without Supervision, TMLR 2024.
    Zhou et al. (2022), iBOT: Image BERT Pre-Training with Online
    Tokenizer, ICLR 2022.
    Sablayrolles et al. (2019), Spreading vectors for similarity
    search, ICLR 2019 (KoLeo).
    """
    S = core.mat(student)
    T = core.mat(teacher)
    if not S or not T:
        raise ValueError("empty input: student and teacher logits are required")
    if len(S) != len(T) or len(S[0]) != len(T[0]):
        raise ValueError("student and teacher must have the same shape")
    if not (tau > 0.0 and tau_t > 0.0):
        raise ValueError("temperatures must be strictly positive")
    P = len(S) - 1
    if P < 0:
        raise ValueError("logits must hold a class token")
    ps = dino_softmax(S[0], tau)
    pt = dino_softmax(T[0], tau_t)
    dino = -sum(pt[k] * math.log(ps[k] + 1e-300) for k in range(len(ps)))
    if mask is None:
        mk = [1 if (i % 2 == 1) else 0 for i in range(P)]
    else:
        mk = [1 if v else 0 for v in core.vec(mask)]
        if len(mk) != P:
            raise ValueError("mask must have one flag per patch")
    ibot = 0.0
    nm = 0
    for i in range(P):
        if not mk[i]:
            continue
        a = dino_softmax(S[1 + i], tau)
        b = dino_softmax(T[1 + i], tau_t)
        ibot += -sum(b[k] * math.log(a[k] + 1e-300) for k in range(len(a)))
        nm += 1
    ibot = ibot / nm if nm else 0.0
    E = core.mat(x)
    n = len(E)
    if n < 2:
        raise ValueError("KoLeo needs at least two embeddings")
    d = len(E[0])
    En = [l2_normalize(r) for r in E]
    koleo = 0.0
    for i in range(n):
        best = float("inf")
        for j in range(n):
            if i == j:
                continue
            dd = math.sqrt(sum((En[i][k] - En[j][k]) ** 2 for k in range(d)))
            if dd < best:
                best = dd
        koleo += -math.log(best + 1e-12)
    koleo /= n
    loss = dino + w_ibot * ibot + w_koleo * koleo
    return RichResult(payload={
        "estimate": loss,
        "loss": loss,
        "dino": dino,
        "ibot": ibot,
        "koleo": koleo,
        "n_masked": nm,
        "n": n,
        "d": d,
        "method": "DINOv2 objective: DINO + iBOT + KoLeo",
    })


def cheatsheet():
    return "dinov2: DINOv2 objective (DINO + iBOT + KoLeo)"


# compact alias per ledger/NAMING.md
dinov2repr = dino_v2_repr
