# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DINO: self-distillation with no labels for visual representation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dino"]


def geron_dino(
    images,
    student,
    teacher,
    center=None,
    tau_s=0.1,
    tau_t=0.04,
    momentum=0.996,
    center_momentum=0.9,
):
    """
    DINO: self-distillation with no labels for visual representation.

    Formula: student matches teacher momentum network on augmented views

    The objective is implemented in full, because every one of its parts
    exists to stop the same failure: with no labels and no negatives, the
    obvious solution is for both networks to output a constant.

    * **Sharpening**: the teacher uses a colder temperature
      (``tau_t < tau_s``), which makes its distribution peaked and gives
      the student something to move towards.
    * **Centering**: a running mean is subtracted from the teacher logits
      before the softmax, which stops any one dimension dominating.
      Sharpening alone collapses to one dimension, centering alone
      collapses to uniform -- together they cancel, and both
      ``teacher_entropy`` and ``kl_to_uniform`` are reported so the
      balance is observable.
    * **Momentum teacher**: the teacher is an EMA of the student,
      ``teacher <- m * teacher + (1-m) * student``, never trained by
      gradient. The updated parameters are returned as
      ``teacher_next``.

    The loss is the cross-entropy between teacher and student
    distributions over *different* views, averaged over all ordered pairs
    with the same view excluded -- matching a view against itself is
    trivially satisfiable.

    ``student`` and ``teacher`` are per-view logits of shape ``(V, K)``,
    or callables applied to ``images``.

    Parameters
    ----------
    images : array-like
        Multi-crop views; passed to callables, otherwise only counted.
    student, teacher : array-like, shape (V, K), or callable
    center : array-like, shape (K,), optional
        Running teacher center; default zeros.
    tau_s : float, default 0.1
    tau_t : float, default 0.04
        Must be smaller than ``tau_s`` -- that is what "sharpening" means.
    momentum : float, default 0.996
        Teacher EMA coefficient in [0, 1).
    center_momentum : float, default 0.9

    Returns
    -------
    result : RichResult
        Keys: loss, per_pair_loss, teacher_probs, student_probs,
        teacher_entropy, kl_to_uniform, center_next, teacher_next,
        n_views, estimate, n, method.

    Examples
    --------
    Two views, two output dimensions. Identical uniform logits give a
    uniform teacher, so the loss is ``log 2`` and the entropy is 1 bit:

    >>> import math
    >>> r = geron_dino(None, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    >>> round(r["loss"], 9) == round(math.log(2), 9)
    True
    >>> round(r["teacher_entropy"], 9)
    0.693147181
    >>> round(r["kl_to_uniform"], 12)
    0.0

    Centering removes a constant offset entirely, which is exactly the
    collapse it is there to prevent:

    >>> a = geron_dino(None, [[0.0, 0.0], [0.0, 0.0]], [[5.0, 5.0], [5.0, 5.0]])
    >>> round(a["loss"], 9) == round(math.log(2), 9)
    True

    A sharp teacher against a student that prefers the other dimension
    costs the full gap -- here ``log(1 + e^50)`` at ``tau_s = 0.1``:

    >>> b = geron_dino(None, [[0.0, 5.0], [0.0, 5.0]], [[10.0, 0.0], [10.0, 0.0]])
    >>> round(b["loss"], 6)
    50.0
    >>> b["teacher_entropy"] < 0.01
    True

    The teacher is an EMA of the student, never trained directly:

    >>> c = geron_dino(None, [[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
    ...                momentum=0.5)
    >>> [round(v, 6) for v in c["teacher_next"][0]]
    [0.5, 0.0]

    A teacher temperature above the student's is rejected:

    >>> geron_dino(None, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], tau_t=0.5)
    Traceback (most recent call last):
      ...
    ValueError: geron_dino: tau_t must be smaller than tau_s for sharpening, got tau_t=0.5 and tau_s=0.1

    References
    ----------
    Géron Ch 16
    """
    S = student(images) if callable(student) else student
    T = teacher(images) if callable(teacher) else teacher
    S = np.atleast_2d(np.asarray(S, dtype=float))
    T = np.atleast_2d(np.asarray(T, dtype=float))
    if S.shape != T.shape:
        raise ValueError(f"geron_dino: student has shape {S.shape} but teacher has shape {T.shape}")
    if S.ndim != 2 or S.shape[0] < 2:
        raise ValueError(f"geron_dino: need at least 2 views of shape (V, K), got {S.shape}")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(T)):
        raise ValueError("geron_dino: student and teacher logits must be finite")
    ts, tt = float(tau_s), float(tau_t)
    if ts <= 0 or tt <= 0:
        raise ValueError(f"geron_dino: temperatures must be positive, got tau_s={ts}, tau_t={tt}")
    if tt >= ts:
        raise ValueError(
            f"geron_dino: tau_t must be smaller than tau_s for sharpening, got tau_t={tt} and tau_s={ts}"
        )
    mom, cmom = float(momentum), float(center_momentum)
    if not (0.0 <= mom < 1.0) or not (0.0 <= cmom < 1.0):
        raise ValueError("geron_dino: momentum and center_momentum must lie in [0, 1)")

    V, K = S.shape
    c = np.zeros(K) if center is None else np.atleast_1d(np.asarray(center, dtype=float))
    if c.size != K:
        raise ValueError(f"geron_dino: center has {c.size} entries but the output width is {K}")

    def softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    Pt = softmax((T - c) / tt)
    Ps = softmax(S / ts)
    logPs = np.log(np.clip(Ps, 1e-30, 1.0))

    pairs = []
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            pairs.append(float(-np.sum(Pt[i] * logPs[j])))
    loss = float(np.mean(pairs))

    ent = float(np.mean(-np.sum(Pt * np.log(np.clip(Pt, 1e-30, 1.0)), axis=1)))
    kl_unif = float(np.mean(np.sum(Pt * (np.log(np.clip(Pt, 1e-30, 1.0)) + np.log(K)), axis=1)))
    c_next = cmom * c + (1 - cmom) * T.mean(axis=0)
    t_next = mom * T + (1 - mom) * S

    return RichResult(
        title="DINO self-distillation",
        summary_lines=[("Loss", loss), ("Views", int(V)), ("Teacher entropy", ent)],
        interpretation="Sharpening and centering pull in opposite directions; together they are what prevents collapse.",
        payload={
            "loss": loss,
            "per_pair_loss": pairs,
            "teacher_probs": Pt.tolist(),
            "student_probs": Ps.tolist(),
            "teacher_entropy": ent,
            "max_entropy": float(np.log(K)),
            "kl_to_uniform": kl_unif,
            "center_next": c_next.tolist(),
            "teacher_next": t_next.tolist(),
            "n_views": int(V),
            "n_pairs": int(len(pairs)),
            "tau_s": ts,
            "tau_t": tt,
            "momentum": mom,
            "estimate": loss,
            "n": int(V),
            "method": "DINO cross-view self-distillation with centering, sharpening and a momentum teacher",
        },
    )


def cheatsheet():
    return "hmdino: DINO: self-distillation with no labels for visual representation"
