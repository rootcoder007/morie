# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt chaining (Alammar Ch 7)."""

from ._richresult import RichResult

__all__ = ["alammar_chain_prompting"]


def alammar_chain_prompting(x, prompts, model):
    """y_k = model(P_k(y_{k-1}, x)): each prompt is a callable
    (previous_output, original_input) -> prompt string, composed in
    order. Every intermediate is returned -- the value of a chain over
    one big prompt is exactly that the intermediates are inspectable.

    References: Alammar and Grootendorst, Ch 7.
    """
    if not callable(model):
        raise ValueError("model must be a callable prompt -> text.")
    ps = list(prompts)
    if not ps:
        raise ValueError("no prompts supplied.")
    y = None
    steps = []
    for i, P in enumerate(ps):
        if not callable(P):
            raise ValueError(f"prompt {i} is not callable.")
        prompt = P(y, x)
        y = str(model(prompt))
        steps.append({"prompt": str(prompt), "output": y})
    return RichResult(payload={
        "final_output": y, "steps": steps,
        "estimate": float(len(steps)), "n": len(steps),
        "method": "Prompt chaining (Alammar Ch 7)"})


def cheatsheet():
    return "alcnp: compose prompts, keep every intermediate inspectable"
