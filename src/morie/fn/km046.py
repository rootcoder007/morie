# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.5: the sentiment PREFIX prompt template."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch3_prefix_prompt_template"]

TEMPLATE = "[x] This movie is [z]"


def _fill_template(template, x, z):
    """Shared slot filler for the Ch 3 templates (Eqs 3.5-3.7).

    Returns (prompt, slot_filled). Imported by km047 and km048 so the
    three templates cannot drift in their slot handling.
    """
    if not isinstance(template, str) or "[x]" not in template:
        raise ValueError("the template must be a string with an [x] slot.")
    if not isinstance(x, str) or not x.strip():
        raise ValueError("x must be a non-empty input string.")
    out = template.replace("[x]", x)
    if z is None:
        return out, False
    if not isinstance(z, str):
        raise ValueError("z must be a string, or None to leave the "
                         "answer slot open.")
    if "[z]" not in template:
        raise ValueError("the template has no [z] answer slot to fill.")
    return out.replace("[z]", z), True


def _result(prompt, filled, eq, template):
    tokens = prompt.split()
    return RichResult(payload={
        "prompt": prompt, "slot_filled": filled, "template": template,
        "tokens": tokens, "estimate": float(len(tokens)), "n": len(tokens),
        "method": f"prompt template (Kamath Eq {eq})"})


def kamath_ch3_prefix_prompt_template(x, z=None, template=TEMPLATE):
    """x' = [x] This movie is [z] -- input and instruction BEFORE the
    answer slot, which is why it is a prefix prompt.

    ``z=None`` leaves the slot open (the usual case: the model fills
    it); passing a string produces the filled prompt.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.5, printed
    p. 100.

    Examples
    --------
    >>> kamath_ch3_prefix_prompt_template("Cannot watch this movie.")["prompt"]
    'Cannot watch this movie. This movie is [z]'
    >>> out = kamath_ch3_prefix_prompt_template("Loved it.", "great")
    >>> out["prompt"], out["slot_filled"]
    ('Loved it. This movie is great', True)
    """
    prompt, filled = _fill_template(template, x, z)
    return _result(prompt, filled, "3.5", template)


def cheatsheet():
    return "km046: prefix template '[x] This movie is [z]'"
