# morie.fn -- function file (rootcoder007/morie)
"""DINO multi-crop student-teacher objective."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["dino_multicrop"]


def dino_softmax(v, tau, center=None):
    n = len(v)
    c = [0.0] * n if center is None else center
    z = [(v[i] - c[i]) / tau for i in range(n)]
    m = max(z)
    e = [math.exp(t - m) for t in z]
    s = sum(e)
    return [t / s for t in e]


def dino_multicrop(image, global_size=2, local_size=8, tau_s=0.1, tau_t=0.04,
                   center=None):
    """
    DINO multi-crop consistency

    Formula: 2 global + 8 local crops; consistency

    Every crop goes through the student, only the GLOBAL crops go
    through the teacher, and the loss is the cross-entropy of the
    student distribution against the sharpened, centred teacher over all
    pairs of different crops.  With V views of which G are global that is
    G(V-1) terms.  Centring plus sharpening is what stops the collapse
    to a constant output.

    Parameters
    ----------
    image : array-like
        V x d matrix of per-crop logits, global crops first.
    global_size : int
        Number of global crops G.
    local_size : int
        Number of local crops; V must equal G + local_size.
    tau_s : float
        Student temperature.
    tau_t : float
        Teacher temperature, smaller than the student's (sharpening).
    center : array-like or None
        Teacher centre vector; None uses the mean of the global logits.

    Returns
    -------
    result : dict
        Keys: estimate (loss), loss, n_pairs, teacher, student_entropy,
        V, d.

    References
    ----------
    Caron et al. (2021), Emerging Properties in Self-Supervised Vision
    Transformers, ICCV 2021:9650-9660.
    """
    M = core.mat(image)
    V = len(M)
    if V == 0:
        raise ValueError("empty input: image has no crops")
    d = len(M[0])
    G = int(global_size)
    L = int(local_size)
    if G < 1:
        raise ValueError("global_size must be at least 1")
    if V != G + L:
        raise ValueError("image must hold global_size + local_size rows")
    if not (tau_s > 0.0 and tau_t > 0.0):
        raise ValueError("temperatures must be strictly positive")
    if center is None:
        c = [sum(M[g][k] for g in range(G)) / G for k in range(d)]
    else:
        c = core.vec(center)
        if len(c) != d:
            raise ValueError("center must have one entry per output dimension")
    teach = [dino_softmax(M[g], tau_t, c) for g in range(G)]
    stud = [dino_softmax(M[v], tau_s) for v in range(V)]
    tot = 0.0
    npair = 0
    for g in range(G):
        for v in range(V):
            if v == g:
                continue
            tot += -sum(teach[g][k] * math.log(stud[v][k] + 1e-300)
                        for k in range(d))
            npair += 1
    loss = tot / npair if npair else float("nan")
    ent = -sum(sum(p * math.log(p + 1e-300) for p in stud[v])
               for v in range(V)) / V
    return RichResult(payload={
        "estimate": loss,
        "loss": loss,
        "n_pairs": npair,
        "teacher": teach,
        "student_entropy": ent,
        "V": V,
        "d": d,
        "method": "DINO multi-crop student-teacher consistency",
    })


def cheatsheet():
    return "dinmlt: DINO multi-crop student-teacher consistency"


# compact alias per ledger/NAMING.md
dinomulticrop = dino_multicrop
