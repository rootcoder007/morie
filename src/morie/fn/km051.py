# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.10: the QA prompt shape carrying adversarial triggers."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_qa_trigger_template"]


def kamath_ch3_qa_trigger_template(x, y, T, z_adv, n_triggers=3):
    """"Question: [x] Context: [y] Answer: [T][T][T][z_adv]".

    ``T`` is the trigger token repeated ``n_triggers`` times (three in
    the book) immediately before the adversarial answer ``z_adv``; the
    gradient-directed search of Wallace et al. optimises exactly those
    repeated slots, so their count is a parameter, not a constant baked
    into a string.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.10, printed
    p. 105.

    Examples
    --------
    >>> out = kamath_ch3_qa_trigger_template(
    ...     "Where?", "Paris.", "the", "Rome")
    >>> out["prompt"]
    'Question: Where? Context: Paris. Answer: the the the Rome'
    >>> out["n_triggers"]
    3
    """
    for name, v in (("x", x), ("y", y), ("T", T), ("z_adv", z_adv)):
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{name} must be a non-empty string.")
    k = int(n_triggers)
    if k < 1:
        raise ValueError("n_triggers must be at least 1; a trigger prompt "
                         "with no triggers is just a QA prompt.")
    triggers = " ".join([T] * k)
    prompt = f"Question: {x} Context: {y} Answer: {triggers} {z_adv}"
    tokens = prompt.split()
    return RichResult(payload={
        "prompt": prompt, "trigger": T, "n_triggers": k,
        "adversarial_answer": z_adv, "tokens": tokens,
        "estimate": float(len(tokens)), "n": len(tokens),
        "method": "adversarial-trigger QA prompt (Kamath Eq 3.10)"})


def cheatsheet():
    return "km051: 'Question: [x] Context: [y] Answer: [T]x3 [z_adv]'"
