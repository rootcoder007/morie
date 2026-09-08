# morie.fn -- function file (rootcoder007/morie)
"""Temperature-scaled knowledge-distillation loss."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["distilkl", "alphazero_distill_student"]


def _softmax(z, T):
    m = max(z)
    e = [math.exp((v - m) / T) for v in z]
    s = sum(e)
    return [v / s for v in e]


def distilkl(teacher, student, temperature=2.0, label=None, alpha=0.5):
    """Soft-target distillation objective at temperature T.

    Class probabilities are produced by a softmax with a temperature,

        p_i = exp(z_i / T) / sum_j exp(z_j / T),                   (eq. 1)

    the teacher's p forming the soft targets.  The distillation objective
    is the cross entropy of the student's soft distribution q against p,
    computed at the same high T.  Because the gradients from the soft
    targets scale as 1/T^2, that term is multiplied by T^2 before being
    combined with the ordinary cross entropy against the hard label,
    which is evaluated at T = 1:

        L = alpha T^2 CE(p, q) + (1 - alpha) CE(onehot, softmax(student))

    Parameters
    ----------
    teacher : array-like
        Teacher logits, length k.
    student : array-like
        Student logits, length k.
    temperature : float
        T > 0.
    label : int or None
        Index of the correct class; ``None`` returns the soft term only.
    alpha : float
        Weight on the soft objective, in [0, 1].

    Returns
    -------
    RichResult
        ``softce``, ``kl``, ``hardce``, ``total``, ``teacherprob``,
        ``studentprob``, ``temperature``, ``k``.

    References
    ----------
    Hinton, G., Vinyals, O. and Dean, J. (2015), "Distilling the knowledge
    in a neural network", arXiv:1503.02531.  Section 2 gives the
    tempered softmax (Equation 1), the two-term objective with a lower
    weight on the hard-label term, and the instruction to multiply the
    soft-target term by T^2 because its gradients scale as 1/T^2.  Read
    from the ar5iv rendering of the arXiv source; the same paper is in the
    local corpus.
    """
    t = C.vec(teacher)
    s = C.vec(student)
    T = float(temperature)
    if len(t) != len(s):
        raise ValueError("teacher and student logits must have equal length")
    if T <= 0.0:
        raise ValueError("temperature must be strictly positive")
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    k = len(t)
    p = _softmax(t, T)
    q = _softmax(s, T)
    ce = -sum(p[i] * math.log(q[i]) for i in range(k))
    kl = sum(p[i] * math.log(p[i] / q[i]) for i in range(k) if p[i] > 0.0)
    hard = float("nan")
    total = T * T * ce
    if label is not None:
        j = int(label)
        if not 0 <= j < k:
            raise ValueError("label out of range")
        q1 = _softmax(s, 1.0)
        hard = -math.log(q1[j])
        total = a * T * T * ce + (1.0 - a) * hard
    return RichResult(payload={
        "softce": ce, "kl": kl, "hardce": hard, "total": total,
        "teacherprob": p, "studentprob": q, "temperature": T, "k": k,
        "method": "Temperature-scaled distillation loss (Hinton et al. 2015 Sect. 2)"})


alphazero_distill_student = distilkl


def cheatsheet():
    return "agdsts: Temperature-scaled knowledge-distillation loss."
