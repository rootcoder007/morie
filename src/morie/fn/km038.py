# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.38: GPT-2's task conditioning p(output|input,task)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_gpt2_task_conditioning"]


def kamath_ch2_gpt2_task_conditioning(input, task, model=None):
    """p(output | input, task): one model, many tasks, the task named
    in the prompt. ``model`` is a callable (input, task) -> named
    distribution over outputs; it is validated to sum to 1 and the
    argmax returned. Without a model the composed prompt alone comes
    back -- honest about what Eq 2.38 is: a definition, not a formula.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.38, printed
    p. 70.

    Examples
    --------
    >>> m = lambda i, t: {"yes": 0.75, "no": 0.25}
    >>> kamath_ch2_gpt2_task_conditioning("ok?", "qa", m)["output"]
    'yes'
    """
    prompt = f"{task}: {input}"
    if model is None:
        return RichResult(payload={
            "prompt": prompt, "output": None, "estimate": 0.0, "n": 0,
            "method": "GPT-2 task conditioning (Kamath Eq 2.38)"})
    dist = model(input, task)
    p = np.array([float(v) for v in dist.values()])
    if np.any(p < 0) or abs(float(p.sum()) - 1.0) > 1e-8:
        raise ValueError(
            "the model's output distribution must be non-negative and "
            f"sum to 1; it sums to {float(p.sum()):.6g}.")
    keys = list(dist.keys())
    best = int(np.argmax(p))
    return RichResult(payload={
        "prompt": prompt, "output": keys[best],
        "distribution": {k: float(v) for k, v in dist.items()},
        "estimate": float(p[best]), "n": len(keys),
        "method": "GPT-2 task conditioning (Kamath Eq 2.38)"})


def cheatsheet():
    return "km038: task-in-prompt conditioning, distribution validated"
