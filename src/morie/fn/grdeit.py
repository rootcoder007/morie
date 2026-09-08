# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DeiT distillation loss: CE on the class token + CE on the distillation token."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_deit_distillation_loss"]

_METHOD = "DeiT hard-label distillation loss"


def _log_softmax(Z):
    M = Z.max(axis=1, keepdims=True)
    Z = Z - M
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def geron_deit_distillation_loss(logits_cls, logits_dist, y, teacher_preds,
                                 alpha=0.5):
    r"""Combine the ground-truth and teacher-agreement objectives.

    .. math::
        L = (1-\alpha)\,\mathrm{CE}(y_{\text{cls}}, y)
          + \alpha\,\mathrm{CE}\bigl(y_{\text{dist}},
            \arg\max \text{teacher}(x)\bigr)

    DeiT's twist is the *separate distillation token*: the two objectives
    get their own output head, so the student is not forced to satisfy
    the label and the teacher through one set of logits.  This is hard
    distillation -- the teacher's argmax, not its soft distribution --
    which is what the paper found worked better and needs no temperature.

    Parameters
    ----------
    logits_cls, logits_dist : array-like, shape (B, C)
        Logits from the class and distillation tokens.
    y : array-like, shape (B,)
        Ground-truth class indices.
    teacher_preds : array-like, shape (B, C) or (B,)
        Teacher logits/probabilities (argmax is taken) or teacher labels.
    alpha : float, optional
        Weight on the distillation term, in ``[0, 1]``. Default 0.5,
        the paper's setting.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``loss_cls``, ``loss_dist``,
        ``teacher_labels``, ``teacher_agreement`` (teacher vs ground
        truth), ``accuracy_cls``, ``accuracy_dist``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 16, DeiT section.

    Examples
    --------
    Uniform logits over two classes give ``log 2`` on both heads:

    >>> import math
    >>> r = geron_deit_distillation_loss([[0.0, 0.0]], [[0.0, 0.0]], [0],
    ...                                  [[1.0, 0.0]])
    >>> round(r["loss"], 6) == round(math.log(2), 6)
    True
    >>> r["teacher_labels"]
    [0]
    >>> r["teacher_agreement"]
    1.0

    When the teacher disagrees with the label the two terms pull apart:

    >>> r2 = geron_deit_distillation_loss([[10.0, 0.0]], [[10.0, 0.0]], [0],
    ...                                   [[0.0, 10.0]])
    >>> f"{r2['loss_cls']:.6f}"
    '0.000045'
    >>> f"{r2['loss_dist']:.6f}"
    '10.000045'
    >>> r2["teacher_agreement"]
    0.0
    """
    Lc = np.atleast_2d(np.asarray(logits_cls, dtype=float))
    Ld = np.atleast_2d(np.asarray(logits_dist, dtype=float))
    if Lc.shape != Ld.shape:
        raise ValueError(
            f"logits_cls shape {Lc.shape} must match logits_dist shape {Ld.shape}."
        )
    if Lc.size == 0:
        raise ValueError("logits are empty.")
    if not np.all(np.isfinite(Lc)) or not np.all(np.isfinite(Ld)):
        raise ValueError("logits must be finite.")
    B, C = Lc.shape
    y = np.asarray(y).ravel().astype(int)
    if y.size != B:
        raise ValueError(f"y must have one label per instance ({B}), got {y.size}.")
    if y.min() < 0 or y.max() >= C:
        raise ValueError(f"y labels must lie in [0, {C - 1}].")

    tp = np.asarray(teacher_preds, dtype=float)
    if tp.ndim == 1:
        t_lab = tp.astype(int)
        if t_lab.size != B:
            raise ValueError(f"teacher_preds must have {B} labels, got {t_lab.size}.")
    elif tp.ndim == 2:
        if tp.shape != (B, C):
            raise ValueError(f"teacher_preds must have shape {(B, C)}, got {tp.shape}.")
        if not np.all(np.isfinite(tp)):
            raise ValueError("teacher_preds contains non-finite values.")
        t_lab = tp.argmax(axis=1)
    else:
        raise ValueError(f"teacher_preds must be 1-D or 2-D, got ndim={tp.ndim}.")
    if t_lab.min() < 0 or t_lab.max() >= C:
        raise ValueError(f"teacher labels must lie in [0, {C - 1}].")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")

    idx = np.arange(B)
    loss_cls = float(-_log_softmax(Lc)[idx, y].mean())
    loss_dist = float(-_log_softmax(Ld)[idx, t_lab].mean())
    loss = (1.0 - alpha) * loss_cls + alpha * loss_dist

    return RichResult(
        title="DeiT distillation loss",
        summary_lines=[("Loss", loss), ("CE (class token)", loss_cls),
                       ("CE (distillation token)", loss_dist)],
        payload={
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_dist": loss_dist,
            "teacher_labels": t_lab.tolist(),
            "teacher_agreement": float(np.mean(t_lab == y)),
            "accuracy_cls": float(np.mean(Lc.argmax(axis=1) == y)),
            "accuracy_dist": float(np.mean(Ld.argmax(axis=1) == t_lab)),
            "alpha": alpha,
            "estimate": loss,
            "n": int(B),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdeit: DeiT loss = (1-a)*CE(cls, y) + a*CE(dist, teacher argmax)"
