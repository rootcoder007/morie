# morie.fn -- slice s03 (rootcoder007/morie)
"""DeiT distillation loss.

Source consulted (FETCHED): Touvron, H., Cord, M., Douze, M., Massa, F.,
Sablayrolles, A. and Jegou, H. (2021).  Training data-efficient image
transformers and distillation through attention.  *ICML* 139, 10347-
10357 (arXiv:2012.12877).  The paper prints both losses.  Soft
distillation, its equation (2):

    L_global = (1 - lambda) L_CE(psi(Z_s), y)
               + lambda tau^2 KL( psi(Z_s / tau), psi(Z_t / tau) )

and hard-label distillation, its equation (3):

    L_global^(hardDistill) = (1/2) L_CE(psi(Z_s), y)
                             + (1/2) L_CE(psi(Z_s), y_t)

with y_t = argmax_c Z_t(c) the teacher's hard decision.  The paper finds
the hard variant works better and uses it for the distillation token, so
``mode="hard"`` is the default; the soft form is available and the tau^2
factor is applied exactly as printed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["deit_distill"]

_EPS = 1e-300


def _ce(logp, target):
    return -logp[int(target)]


def deit_distill(x, teacher=None, y=None, mode="hard", lam=0.5, tau=1.0):
    """DeiT distillation loss for one example.

    Parameters
    ----------
    x : array-like
        Student logits Z_s.
    teacher : array-like
        Teacher logits Z_t.
    y : int
        The true label.
    mode : {"hard", "soft"}
        Which of the paper's two losses to compute.
    lam : float
        The soft-distillation lambda.
    tau : float
        The distillation temperature.

    Returns
    -------
    RichResult with payload:
        estimate : the total loss
        ce, kd   : its two parts
        y_teacher: the teacher's hard decision
    """
    zs = k.vec(x)
    zt = k.vec(teacher) if teacher is not None else []
    ps = k.softmax(zs)
    logps = [math.log(p if p > _EPS else _EPS) for p in ps]
    yy = int(y) if y is not None else 0
    ce = _ce(logps, yy)
    yt = 0
    for i in range(1, len(zt)):
        if zt[i] > zt[yt]:
            yt = i
    if mode == "soft":
        t = float(tau)
        pst = k.softmax([z / t for z in zs])
        ptt = k.softmax([z / t for z in zt])
        kl = 0.0
        for i in range(len(ptt)):
            if ptt[i] > 0.0:
                kl += ptt[i] * (math.log(ptt[i])
                                - math.log(pst[i] if pst[i] > _EPS else _EPS))
        kd = float(lam) * t * t * kl
        total = (1.0 - float(lam)) * ce + kd
    else:
        kd = 0.5 * _ce(logps, yt)
        total = 0.5 * ce + kd
    return RichResult(
        title="DeiT distillation loss",
        summary_lines=[("loss", total), ("mode", mode)],
        payload={
            "estimate": total,
            "ce": ce,
            "kd": kd,
            "y_teacher": yt,
            "mode": mode,
            "method": "DeiT distillation loss (Touvron et al. 2021, eqs. 2-3)",
        },
    )


def cheatsheet():
    return "deitsr: DeiT distillation token"


deitdistill = deit_distill
