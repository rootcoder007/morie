# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DINO self-distillation loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dino_self_distillation"]

_METHOD = "DINO self-distillation cross-entropy"


def _log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = z.max(axis=-1, keepdims=True)
    z = z - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def geron_dino_self_distillation(student_logits, teacher_logits, tau_s, tau_t, center=None):
    r"""Student matches a sharpened, centred teacher distribution.

    .. math::
        L = -\sum_k P_{\text{teacher}}(k)\,\log P_{\text{student}}(k),
        \qquad
        P_{\text{teacher}} = \mathrm{softmax}\!\bigl((t - c)/\tau_t\bigr),
        \quad P_{\text{student}} = \mathrm{softmax}(s/\tau_s)

    There is no label anywhere -- the teacher is an EMA of the student,
    and the gradient flows only through the student (stop-grad on the
    teacher side), which is exactly what this function computes: the
    teacher's distribution enters as a constant weight.

    Sharpening (:math:`\tau_t < \tau_s`) and centering are the two
    halves of DINO's collapse prevention, and they push in opposite
    directions.  Sharpening drives the teacher toward one-hot;
    centering, subtracting a running mean, stops any single dimension
    from dominating.  Drop either and the network collapses to a
    constant output -- which the reported ``teacher_entropy`` detects:
    it heads for 0 under collapse to one dimension, and for
    :math:`\log K` under uniform collapse.

    Parameters
    ----------
    student_logits, teacher_logits : array-like, shape (K,) or (m, K)
    tau_s, tau_t : float
        Student and teacher temperatures, positive. DINO uses
        ``tau_t < tau_s`` (0.04 vs 0.1).
    center : array-like, shape (K,), optional
        Running mean subtracted from the teacher logits. Default none.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``teacher_probs``, ``student_probs``,
        ``teacher_entropy``, ``student_entropy``, ``max_teacher_prob``,
        ``is_sharpened`` (``tau_t < tau_s``), ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 16, DINO section (Caron et al. 2021).

    Examples
    --------
    Two flat distributions cost ``log 2`` -- the cross-entropy of
    uniform with uniform:

    >>> r = geron_dino_self_distillation([0.0, 0.0], [0.0, 0.0], 0.1, 0.04)
    >>> round(r["loss"], 10)
    0.6931471806
    >>> r["is_sharpened"]
    True

    The teacher's low temperature sharpens it far past its raw logits:
    logits ``[0.4, 0]`` at ``tau_t = 0.04`` are almost one-hot.

    >>> r2 = geron_dino_self_distillation([0.0, 0.0], [0.4, 0.0], 0.1, 0.04)
    >>> round(r2["max_teacher_prob"], 6)
    0.999955

    Centering shifts the teacher, and an equal shift on both logits
    changes nothing (softmax is shift-invariant):

    >>> r3 = geron_dino_self_distillation([0.0, 0.0], [0.4, 0.0], 0.1, 0.04,
    ...                                    center=[1.0, 1.0])
    >>> round(r3["max_teacher_prob"], 6)
    0.999955
    """
    S = np.atleast_2d(np.asarray(student_logits, dtype=float))
    Tl = np.atleast_2d(np.asarray(teacher_logits, dtype=float))
    if S.shape != Tl.shape:
        raise ValueError(f"student_logits {S.shape} and teacher_logits {Tl.shape} must match.")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(Tl)):
        raise ValueError("logits must be finite.")
    ts, tt = float(tau_s), float(tau_t)
    for name, v in (("tau_s", ts), ("tau_t", tt)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a positive finite temperature, got {v}.")
    m, K = S.shape
    if center is not None:
        c = np.asarray(center, dtype=float).ravel()
        if c.size != K:
            raise ValueError(f"center must have {K} entries, got {c.size}.")
        if not np.all(np.isfinite(c)):
            raise ValueError("center must be finite.")
        Tl = Tl - c

    logp_t = _log_softmax(Tl / tt)
    logp_s = _log_softmax(S / ts)
    P = np.exp(logp_t)
    Q = np.exp(logp_s)
    per = -np.sum(P * logp_s, axis=1)
    loss = float(per.mean())

    return RichResult(
        title="DINO self-distillation",
        summary_lines=[("Loss", loss), ("tau_s / tau_t", f"{ts} / {tt}"),
                       ("Teacher entropy", float(-np.sum(P * logp_t, axis=1).mean()))],
        payload={
            "loss": loss,
            "teacher_probs": P[0].tolist() if m == 1 else P.tolist(),
            "student_probs": Q[0].tolist() if m == 1 else Q.tolist(),
            "teacher_entropy": float(-np.sum(P * logp_t, axis=1).mean()),
            "student_entropy": float(-np.sum(Q * logp_s, axis=1).mean()),
            "max_teacher_prob": float(P.max(axis=1).mean()),
            "is_sharpened": bool(tt < ts),
            "per_sample_loss": per.tolist(),
            "estimate": loss,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdino: -sum P_teacher log P_student, teacher sharpened (tau_t < tau_s) and centred"
