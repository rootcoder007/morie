# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 3.7: the sentiment CLOZE prompt template."""

from . import _array_core as np

from ._richresult import RichResult
from .km046 import _fill_template, _result

__all__ = ["kamath_ch3_cloze_prompt_template"]

TEMPLATE = "[x] This is a [z] movie."


def kamath_ch3_cloze_prompt_template(x, z=None, template=TEMPLATE):
    """x' = [x] This is a [z] movie. -- template tokens on BOTH sides
    of the answer slot, which is what makes it a cloze rather than a
    prefix prompt. Slot filling delegates to km046's shared filler.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 3, Eq 3.7, printed
    p. 101.

    Examples
    --------
    >>> kamath_ch3_cloze_prompt_template("Cannot watch this.")["prompt"]
    'Cannot watch this. This is a [z] movie.'
    >>> out = kamath_ch3_cloze_prompt_template("Loved it.", "great")
    >>> out["prompt"]
    'Loved it. This is a great movie.'
    """
    prompt, filled = _fill_template(template, x, z)
    return _result(prompt, filled, "3.7", template)


def cheatsheet():
    return "km048: cloze template '[x] This is a [z] movie.'"
