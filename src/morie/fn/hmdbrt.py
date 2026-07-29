# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DistilBERT: distilled BERT with ~40% fewer parameters."""

import numpy as np

from ._richresult import RichResult
from .hmencox import geron_encoder_only

__all__ = ["geron_distilbert"]


def geron_distilbert(teacher, student, X, temperature=2.0, alpha_ce=0.5, alpha_mlm=0.5, alpha_cos=0.0, mlm_labels=None):
    """
    DistilBERT: distilled BERT with ~40% fewer parameters.

    Formula: student mimics teacher outputs + masked LM

    The triple loss is computed for real:

    * **distillation**, ``T^2 * KL(teacher_T || student_T)`` -- the
      ``T^2`` factor is not decoration, it restores the gradient
      magnitude that dividing the logits by ``T`` removed, so the term
      keeps its weight as the temperature rises;
    * **masked language modelling**, ordinary cross-entropy against
      ``mlm_labels`` at temperature 1;
    * **cosine embedding**, ``1 - cos(h_s, h_t)``, aligning hidden
      directions rather than only output probabilities.

    ``teacher`` and ``student`` are logits arrays or callables applied to
    ``X``. When they carry hidden states as ``(logits, hidden)`` the
    cosine term is available.

    The compression claim is checked against the architectures rather
    than repeated: BERT-base (12 layers) and DistilBERT (6 layers) are
    resolved through :func:`morie.fn.hmencox.geron_encoder_only` and the
    real ratio is returned as ``param_reduction``. It is not 50% despite
    halving the depth, because the embedding table is shared and does not
    shrink -- which is exactly why the paper says 40%.

    Parameters
    ----------
    teacher, student : array-like, shape (B, C), tuple, or callable
        Logits, or ``(logits, hidden)``.
    X : array-like
        Inputs passed to callables; also used for the token count.
    temperature : float, default 2.0
        Distillation temperature, >= 1.
    alpha_ce, alpha_mlm, alpha_cos : float
        Non-negative weights that must not all be zero.
    mlm_labels : array-like, optional
        Required when ``alpha_mlm > 0``.

    Returns
    -------
    result : RichResult
        Keys: loss, loss_ce, loss_mlm, loss_cos, teacher_params,
        student_params, param_reduction, agreement, estimate, n, method.

    Examples
    --------
    A student that matches its teacher exactly pays nothing on the
    distillation term:

    >>> r = geron_distilbert([[2.0, 0.0]], [[2.0, 0.0]], [1, 2, 3],
    ...                      alpha_mlm=0.0, alpha_ce=1.0)
    >>> round(r["loss_ce"], 12)
    0.0
    >>> r["agreement"]
    1.0

    Disagreement costs, and the ``T^2`` factor keeps the term from
    vanishing as ``T`` grows:

    >>> a = geron_distilbert([[10.0, 0.0]], [[0.0, 0.0]], [1], alpha_mlm=0.0,
    ...                      alpha_ce=1.0, temperature=1.0)
    >>> b = geron_distilbert([[10.0, 0.0]], [[0.0, 0.0]], [1], alpha_mlm=0.0,
    ...                      alpha_ce=1.0, temperature=2.0)
    >>> round(a["loss_ce"], 6)
    0.692648
    >>> round(b["loss_ce"], 6)
    2.61187

    The MLM term is ordinary cross-entropy; uniform logits cost ``log 2``:

    >>> import math
    >>> r2 = geron_distilbert([[0.0, 0.0]], [[0.0, 0.0]], [1], alpha_ce=0.0,
    ...                       alpha_mlm=1.0, mlm_labels=[0])
    >>> round(r2["loss_mlm"], 9) == round(math.log(2), 9)
    True

    The compression is measured, not assumed:

    >>> round(r["param_reduction"], 3)
    0.391

    References
    ----------
    Géron Ch 15
    """
    T = float(temperature)
    if not np.isfinite(T) or T < 1:
        raise ValueError(f"geron_distilbert: temperature must be >= 1, got {temperature!r}")
    a_ce, a_mlm, a_cos = float(alpha_ce), float(alpha_mlm), float(alpha_cos)
    if min(a_ce, a_mlm, a_cos) < 0:
        raise ValueError("geron_distilbert: the loss weights must be non-negative")
    if a_ce + a_mlm + a_cos <= 0:
        raise ValueError("geron_distilbert: at least one loss weight must be positive")

    def unpack(m, name):
        v = m(X) if callable(m) else m
        if isinstance(v, tuple) and len(v) == 2:
            return np.atleast_2d(np.asarray(v[0], dtype=float)), np.atleast_2d(np.asarray(v[1], dtype=float))
        return np.atleast_2d(np.asarray(v, dtype=float)), None

    Zt, Ht = unpack(teacher, "teacher")
    Zs, Hs = unpack(student, "student")
    if Zt.shape != Zs.shape:
        raise ValueError(f"geron_distilbert: teacher logits {Zt.shape} do not match student logits {Zs.shape}")
    if Zt.size == 0:
        raise ValueError("geron_distilbert: no logits supplied")
    if not np.all(np.isfinite(Zt)) or not np.all(np.isfinite(Zs)):
        raise ValueError("geron_distilbert: logits must be finite")
    B, C = Zt.shape

    def logsoftmax(Z, temp):
        Zn = Z / temp
        Zn = Zn - Zn.max(axis=1, keepdims=True)
        return Zn - np.log(np.exp(Zn).sum(axis=1, keepdims=True))

    lpt = logsoftmax(Zt, T)
    lps = logsoftmax(Zs, T)
    pt = np.exp(lpt)
    loss_ce = float(T * T * np.mean(np.sum(pt * (lpt - lps), axis=1)))

    loss_mlm = 0.0
    if a_mlm > 0:
        if mlm_labels is None:
            raise ValueError("geron_distilbert: alpha_mlm > 0 requires mlm_labels")
        y = np.asarray(mlm_labels).ravel().astype(int)
        if y.size != B:
            raise ValueError(f"geron_distilbert: mlm_labels has {y.size} entries but there are {B} rows")
        if y.min() < 0 or y.max() >= C:
            raise ValueError(f"geron_distilbert: an MLM label lies outside 0..{C - 1}")
        lp1 = logsoftmax(Zs, 1.0)
        loss_mlm = float(-np.mean(lp1[np.arange(B), y]))

    loss_cos = 0.0
    if a_cos > 0:
        if Hs is None or Ht is None:
            raise ValueError("geron_distilbert: alpha_cos > 0 requires hidden states as (logits, hidden)")
        if Hs.shape != Ht.shape:
            raise ValueError(f"geron_distilbert: hidden states differ in shape, {Hs.shape} vs {Ht.shape}")
        ns = np.linalg.norm(Hs, axis=1)
        nt = np.linalg.norm(Ht, axis=1)
        if np.any(ns == 0) or np.any(nt == 0):
            raise ValueError("geron_distilbert: a hidden state is the zero vector, so its cosine is undefined")
        loss_cos = float(np.mean(1.0 - np.sum(Hs * Ht, axis=1) / (ns * nt)))

    total = a_ce * loss_ce + a_mlm * loss_mlm + a_cos * loss_cos

    tokens = int(np.asarray(X).ravel().size) if X is not None else 1
    teach_arch = geron_encoder_only(list(range(max(tokens, 1))), n_layers=12)
    stud_arch = geron_encoder_only(list(range(max(tokens, 1))), n_layers=6)
    tp, sp = int(teach_arch["total_params"]), int(stud_arch["total_params"])

    return RichResult(
        title="DistilBERT distillation",
        summary_lines=[("Loss", total), ("Parameter reduction", float(1 - sp / tp)), ("Temperature", T)],
        interpretation="Halving the depth removes only ~40% of the parameters, because the embedding table is unchanged.",
        payload={
            "loss": total,
            "loss_ce": loss_ce,
            "loss_mlm": loss_mlm,
            "loss_cos": loss_cos,
            "teacher_params": tp,
            "student_params": sp,
            "param_reduction": float(1 - sp / tp),
            "agreement": float(np.mean(Zt.argmax(axis=1) == Zs.argmax(axis=1))),
            "temperature": T,
            "weights": {"ce": a_ce, "mlm": a_mlm, "cos": a_cos},
            "estimate": total,
            "n": int(B),
            "method": "DistilBERT triple loss with T^2-scaled KL; architectures resolved through hmencox",
        },
    )


def cheatsheet():
    return "hmdbrt: DistilBERT: distilled BERT with ~40% fewer parameters"
