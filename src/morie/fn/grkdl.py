# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Knowledge distillation loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_knowledge_distillation_loss"]

_METHOD = "Knowledge distillation (hard CE + soft KL)"


def _log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max(axis=-1, keepdims=True)
    z = z - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def geron_knowledge_distillation_loss(student_logits, teacher_logits, y, alpha, T):
    r"""Blend the hard-label loss with the teacher's soft targets.

    .. math::
        L = (1-\alpha)\,\mathrm{CE}(\text{student}, y)
        + \alpha\, T^2\, \mathrm{KL}\bigl(
        \mathrm{softmax}(s/T)\,\|\,\mathrm{softmax}(t/T)\bigr)

    The temperature is where the information is.  At ``T = 1`` a
    confident teacher's distribution is nearly one-hot and says little
    the label did not; raising ``T`` flattens it and exposes the "dark
    knowledge" -- that a 7 looks somewhat like a 1 and not at all like a
    5.  The ``T^2`` factor restores the gradient magnitude, which
    softening otherwise shrinks by :math:`1/T^2`, so ``alpha`` keeps
    meaning the same thing as ``T`` changes.

    Note the KL direction follows the formula as given,
    ``KL(student || teacher)``.  Hinton's original writes the
    cross-entropy of the teacher's soft targets under the student, i.e.
    ``KL(teacher || student)``; the two differ whenever the
    distributions differ, so both are reported in the payload
    (``kl_student_teacher``, ``kl_teacher_student``).

    Parameters
    ----------
    student_logits, teacher_logits : array-like, shape (K,) or (m, K)
    y : array-like of int, shape (m,) or int
        Hard labels.
    alpha : float
        Soft-loss weight in ``[0, 1]``.
    T : float
        Temperature, positive.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``ce_hard``, ``kl_soft``,
        ``kl_student_teacher``, ``kl_teacher_student``,
        ``soft_targets``, ``teacher_entropy``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 17, Knowledge Distillation section (Hinton et al. 2015).

    Examples
    --------
    A student that matches the teacher exactly pays no soft loss at all,
    so only the hard term survives -- here ``0.5 * log 2``:

    >>> r = geron_knowledge_distillation_loss([0.0, 0.0], [0.0, 0.0], 0,
    ...                                        alpha=0.5, T=2.0)
    >>> round(r["kl_soft"], 12)
    0.0
    >>> round(r["ce_hard"], 10), round(r["loss"], 10)
    (0.6931471806, 0.3465735903)

    Temperature really does soften: at ``T = 4`` a teacher with logits
    ``[4, 0]`` puts 0.73 on the top class instead of 0.98.

    >>> r2 = geron_knowledge_distillation_loss([0.0, 0.0], [4.0, 0.0], 0,
    ...                                         alpha=1.0, T=4.0)
    >>> [round(p, 6) for p in r2["soft_targets"]]
    [0.731059, 0.268941]
    >>> r3 = geron_knowledge_distillation_loss([0.0, 0.0], [4.0, 0.0], 0,
    ...                                         alpha=1.0, T=1.0)
    >>> [round(p, 6) for p in r3["soft_targets"]]
    [0.982014, 0.017986]
    """
    S = np.atleast_2d(np.asarray(student_logits, dtype=float))
    Tl = np.atleast_2d(np.asarray(teacher_logits, dtype=float))
    if S.shape != Tl.shape:
        raise ValueError(f"student_logits {S.shape} and teacher_logits {Tl.shape} must match.")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(Tl)):
        raise ValueError("logits must be finite.")
    m, K = S.shape
    lab = np.atleast_1d(np.asarray(y)).ravel().astype(int)
    if lab.size != m:
        raise ValueError(f"y has {lab.size} entries but there are {m} instances.")
    if lab.min() < 0 or lab.max() >= K:
        raise ValueError(f"y must lie in [0, {K - 1}], got range [{lab.min()}, {lab.max()}].")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")
    Temp = float(T)
    if not np.isfinite(Temp) or Temp <= 0:
        raise ValueError(f"T must be a positive finite temperature, got {Temp}.")

    logp_hard = _log_softmax(S)
    ce = float(-logp_hard[np.arange(m), lab].mean())

    logq = _log_softmax(S / Temp)          # student, softened
    logt = _log_softmax(Tl / Temp)         # teacher, softened
    q, p = np.exp(logq), np.exp(logt)
    kl_st = float(np.sum(q * (logq - logt), axis=1).mean())
    kl_ts = float(np.sum(p * (logt - logq), axis=1).mean())
    soft = Temp**2 * kl_st
    loss = (1.0 - alpha) * ce + alpha * soft
    ent = float(-np.sum(p * logt, axis=1).mean())

    return RichResult(
        title="Knowledge distillation loss",
        summary_lines=[("Loss", loss), ("CE (hard)", ce),
                       ("KL (soft, x T^2)", soft), ("T", Temp)],
        payload={
            "loss": loss,
            "ce_hard": ce,
            "kl_soft": soft,
            "kl_student_teacher": kl_st,
            "kl_teacher_student": kl_ts,
            "soft_targets": p[0].tolist() if m == 1 else p.tolist(),
            "teacher_entropy": ent,
            "alpha": alpha,
            "T": Temp,
            "estimate": loss,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkdl: (1-a)CE(s,y) + a T^2 KL(soft student || soft teacher); T^2 restores the gradient"
