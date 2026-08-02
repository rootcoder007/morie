# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Continued MLM pretraining before task fine-tuning
(Gururangan et al. 2020; Alammar Ch 11)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_continued_pretraining_mlm"]


def alammar_continued_pretraining_mlm(domain_corpus, mlm_loss_fn,
                                      n_mlm_steps, task_loss_fn=None):
    """Phase 1: n steps of the MLM loss on the domain corpus; phase 2:
    the task loss. ``mlm_loss_fn`` is (corpus, step) -> loss, so the
    caller's model closes over its own state; the loss CURVE comes
    back, and the payload reports whether it decreased -- domain
    adaptation that does not reduce the domain loss did nothing.

    References: Alammar and Grootendorst, Ch 11; Gururangan et al.
    (2020).
    """
    if not callable(mlm_loss_fn):
        raise ValueError("mlm_loss_fn must be callable (corpus, step) "
                         "-> loss.")
    steps = int(n_mlm_steps)
    if steps < 1:
        raise ValueError("n_mlm_steps must be positive.")
    docs = list(domain_corpus)
    if not docs:
        raise ValueError("the domain corpus is empty.")
    curve = [float(mlm_loss_fn(docs, s)) for s in range(steps)]
    task_loss = float(task_loss_fn()) if callable(task_loss_fn) else None
    return RichResult(payload={
        "mlm_loss_curve": curve,
        "mlm_improved": curve[-1] < curve[0] if steps > 1 else None,
        "task_loss": task_loss,
        "estimate": curve[-1], "n": steps,
        "method": "Continued domain pretraining (Gururangan et al. 2020)"})


def cheatsheet():
    return "alcont: MLM loss curve on domain corpus, improvement reported"
